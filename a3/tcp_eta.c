/*
 * tcp_eta.c — Custom TCP Congestion Control: η-based algorithm
 *
 * Derived from ML observations (Assignment 2) of Linux CUBIC behavior.
 * Implements congestion avoidance with:
 *   - RTT-based preemptive backoff (unlike CUBIC which waits for loss)
 *   - 0.7× multiplicative decrease on loss (same as CUBIC, gentler than Reno's 0.5×)
 *   - Concave recovery toward W_max (gap-closing, CUBIC-inspired)
 *   - η-gated additive increase: only probe when reward signal is positive
 *
 * Build:  make
 * Load:   sudo insmod tcp_eta.ko
 * Verify: cat /proc/sys/net/ipv4/tcp_available_congestion_control
 * Select: setsockopt(fd, IPPROTO_TCP, TCP_CONGESTION, "tcp_eta", 8)
 * Unload: sudo rmmod tcp_eta
 *
 * Algorithm (per-RTT update):
 *   η = goodput - α·RTT - β·loss
 *   if loss:         cwnd = max(2, 0.7 × cwnd)   [save W_max first]
 *   elif RTT > 1.25×rtt_base: cwnd = max(2, 0.9 × cwnd)
 *   elif cwnd < W_max and η>0: cwnd += max(1, (W_max-cwnd)/8)
 *   elif η > 0:      cwnd += 1
 *   else:            hold (no change)
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <net/tcp.h>

/* Hyper-parameters */
#define RTT_THRESH_NUM   2      /* RTT backoff threshold: 2.0× baseline */
#define RTT_THRESH_DEN   1
#define LOSS_NUM         7      /* MD on loss: 0.7× cwnd */
#define LOSS_DEN        10
#define RTT_BACK_NUM     9      /* RTT backoff: 0.9× cwnd */
#define RTT_BACK_DEN    10
#define RECOVER_DIV      8      /* concave recovery step: gap/8 per RTT */
#define EWMA_SHIFT     100      /* rtt_base EWMA weight: 1/100 */
#define MIN_CWND         2
#define BETA            10      /* β loss penalty */

/* Per-connection state (stored in icsk_ca_priv) */
struct eta_ca {
	u32 w_max;               /* cwnd saved at last loss event (W_max) */
	u32 rtt_base_us;         /* minimum RTT baseline, EWMA-updated     */
	u32 loss_prev;           /* tp->total_retrans snapshot for delta    */
	u32 next_rtt_delivered;  /* tp->delivered gate for once-per-RTT    */
};

static inline struct eta_ca *eta_ca(struct sock *sk)
{
	return inet_csk_ca(sk);
}

static void eta_update_rtt_base(struct eta_ca *ca, u32 rtt_us)
{
	if (ca->rtt_base_us == 0 || rtt_us < ca->rtt_base_us)
		ca->rtt_base_us = rtt_us;
	else
		ca->rtt_base_us += (rtt_us - ca->rtt_base_us) / EWMA_SHIFT;
}

static void eta_init(struct sock *sk)
{
	struct eta_ca *ca = eta_ca(sk);

	ca->w_max              = 0;
	ca->rtt_base_us        = 0;
	ca->loss_prev          = 0;
	ca->next_rtt_delivered = 0;
}

static u32 eta_ssthresh(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct eta_ca   *ca = eta_ca(sk);

	ca->w_max              = tp->snd_cwnd;
	ca->loss_prev          = tp->total_retrans;
	ca->next_rtt_delivered = tp->delivered;

	return max_t(u32, MIN_CWND,
		     (tp->snd_cwnd * LOSS_NUM) / LOSS_DEN);
}

static void eta_cong_avoid(struct sock *sk, u32 ack, u32 acked)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct eta_ca   *ca = eta_ca(sk);
	u32 rtt_us, loss_delta, cwnd;
	s64 eta;

	if (!tcp_is_cwnd_limited(sk))
		return;

	/* Slow start: kernel exponential growth (same as Reno/CUBIC) */
	if (tcp_in_slow_start(tp)) {
		tcp_slow_start(tp, acked);
		return;
	}

	/* ---- per-RTT gate (CA phase only) ---- */
	if (before(tp->delivered, ca->next_rtt_delivered))
		return;

	/* Advance gate by one cwnd worth of delivered packets */
	ca->next_rtt_delivered = tp->delivered + tp->snd_cwnd;

	rtt_us = tp->srtt_us >> 3;
	if (rtt_us == 0)
		rtt_us = 1;

	loss_delta = tp->total_retrans - ca->loss_prev;
	ca->loss_prev = tp->total_retrans;
	eta_update_rtt_base(ca, rtt_us);

	/* η = goodput_kbps - α·rtt_kbps - β·loss  (all in kbps, integer arithmetic)
	 * α=0.0001: goodput_kbps = cwnd×MSS×8000/rtt_us; alpha = rtt_us/10000 */
	cwnd = tp->snd_cwnd;
	{
		u64 goodput_kbps = (u64)cwnd * tp->mss_cache * 8000ULL / rtt_us;
		u64 alpha_kbps   = (u64)rtt_us / 10000ULL;
		u64 beta_kbps    = (u64)BETA * loss_delta * 1000ULL;
		eta = (s64)goodput_kbps - (s64)alpha_kbps - (s64)beta_kbps;
	}

	if (ca->rtt_base_us > 0 &&
	    rtt_us * RTT_THRESH_DEN > ca->rtt_base_us * RTT_THRESH_NUM) {
		/* RTT > 2×baseline: preemptive backoff */
		cwnd = max_t(u32, MIN_CWND, (cwnd * RTT_BACK_NUM) / RTT_BACK_DEN);
	} else if (cwnd < ca->w_max && eta > 0) {
		/* Concave recovery toward W_max */
		cwnd += max_t(u32, 1, (ca->w_max - cwnd) / RECOVER_DIV);
	} else if (eta > 0) {
		cwnd += 1;
	}
	/* else: hold — decreases via ssthresh (loss) or RTT branch above */

	tp->snd_cwnd = min_t(u32, cwnd, tp->snd_cwnd_clamp);
}

static u32 eta_undo_cwnd(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct eta_ca   *ca = eta_ca(sk);

	return max_t(u32, tp->snd_cwnd, ca->w_max);
}

static void eta_set_state(struct sock *sk, u8 new_state)
{
	struct eta_ca *ca = eta_ca(sk);

	if (new_state == TCP_CA_Loss) {
		ca->w_max       = 0;
		ca->rtt_base_us = 0;
	}
}

static struct tcp_congestion_ops tcp_eta_ops __read_mostly = {
	.name		= "tcp_eta",
	.owner		= THIS_MODULE,
	.init		= eta_init,
	.ssthresh	= eta_ssthresh,
	.cong_avoid	= eta_cong_avoid,
	.undo_cwnd	= eta_undo_cwnd,
	.set_state	= eta_set_state,
};

static int __init tcp_eta_init(void)
{
	int ret = tcp_register_congestion_control(&tcp_eta_ops);

	if (ret)
		pr_err("tcp_eta: failed to register congestion control: %d\n", ret);
	else
		pr_info("tcp_eta: registered (η-based CC, RTT-aware)\n");

	return ret;
}

static void __exit tcp_eta_exit(void)
{
	tcp_unregister_congestion_control(&tcp_eta_ops);
	pr_info("tcp_eta: unregistered\n");
}

module_init(tcp_eta_init);
module_exit(tcp_eta_exit);

MODULE_AUTHOR("CS53600");
MODULE_DESCRIPTION("η-based TCP Congestion Control (Assignment 3)");
MODULE_LICENSE("GPL");
