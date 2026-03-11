#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif



static const char ____versions[]
__used __section("__versions") =
	"\x28\x00\x00\x00\xa8\x3f\x07\x0d"
	"tcp_register_congestion_control\0"
	"\x10\x00\x00\x00\x7e\x3a\x2c\x12"
	"_printk\0"
	"\x2c\x00\x00\x00\xf3\x64\x8c\xb0"
	"tcp_unregister_congestion_control\0\0\0"
	"\x18\x00\x00\x00\x92\xc7\x05\xd0"
	"tcp_slow_start\0\0"
	"\x18\x00\x00\x00\xd4\xce\x9f\xb7"
	"module_layout\0\0\0"
	"\x00\x00\x00\x00\x00\x00\x00\x00";

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "5E73FED49E64168DCA49ED8");
