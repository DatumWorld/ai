## Ubuntu和Windows双系统从mbr转换gpt后修复引导操作

### 1 windows下 diskgenius 将磁盘分区表从mbr转换 gpt
### 2 修复ubuntu引导。
使用boot-repair-disk-64bit.iso （ https://sourceforge.net/projects/boot-repair-cd/ ） 制作启动盘修复Ubuntu引导

根据提示指定或创建分区操作，可能需要使用Gparted灯类似工具指定或创建，然后根据提示执行命令等，可以修复ubuntu引导

### 3 使用windows安装U盘命令行修复windows引导
Windows 可能还在找它的旧 MBR 启动文件，而 GPT 分区表需要的是 EFI 系统分区（ESP）里的启动文件。这种情况 os-prober 往往扫描不到，因为磁盘上根本没有有效的 Windows EFI 引导文件。需要手动为 Windows 重建 UEFI 启动项。

修复步骤：
1. 确认是否有 EFI 分区

在 Ubuntu 终端输入：

~~~bash
lsblk -f
~~~
查看是否有分区格式为 vfat 且标签通常为 EFI 或 SYSTEM 的分区（一般大小为 100MB-500MB）。

如果没有： 你需要用磁盘工具（如 GParted）压缩出 200MB 空间，新建一个 fat32 分区，并加上 esp 标志。
2. 使用 Windows 环境重建引导（最有效）

由于转换了分区表，必须把 Windows 的启动模式从 BIOS 切换到 UEFI。这需要借助 Windows 安装 U 盘：

从 Windows 安装 U 盘 启动，按 Shift + F10 打开命令提示符（理论上同一版本的Windows即可，实测win11 ltsc安装U盘可修复 2025的引导）。

输入 diskpart 进入磁盘工具：
```
list disk（确认你的硬盘编号，假设为 0）
sel disk 0
list vol（找到那个 FAT32 格式的 EFI 分区，假设它是 Volume 3）
sel vol 3
assign letter=S（给它分配个盘符 S）
exit
修复引导文件（核心步骤）：
假设你的 Windows 系统盘现在是 C:，输入：
cmd
bcdboot C:\Windows /s S: /f UEFI

这行命令会将 UEFI 引导文件从系统盘复制到刚才分配的 S 盘（EFI 分区）中。
```
如何确定哪个是系统分区？
```
在命令提示符中依次输入以下命令，查看哪个盘下有 Windows 文件夹：
输入 dir C: 并回车
输入 dir D: 并回车
输入 dir E: 并回车
```
判断标准：
如果返回的内容中包含 Windows、Users、Program Files 这几个文件夹，那么该盘符就是你的系统盘。例如，如果你在执行 dir D: 时看到了这些文件夹，那么在修复命令中就要用 D:\Windows。


3. 回到 Ubuntu 更新 GRUB
修复完 Windows 引导后，重启进入 Ubuntu，再次执行：
bash
sudo update-grub

此时 os-prober 应该能识别出 Windows Boot Manager 了。

### 4 Windows更新可能删除ubuntu引导，
```
bcdedit /set {bootmgr} path \EFI\ubuntu\grubx64.efi
```
将windows引导挂在ubuntu下面，别让它抢跑.

### 5 如果不行，可能引导文件被windows搞乱了

上安装u盘，启用终极大法（先备份重要数据，再重装，建议先安装win，再安装ubuntu）。