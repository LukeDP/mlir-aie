; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie"

@a14 = external global [256 x i32]

declare void @debug_i32(i32)

; Unknown intrinsic
declare void @llvm.aie.event0()

; Unknown intrinsic
declare void @llvm.aie.event1()

; Unknown intrinsic
declare void @llvm.aie.put.ms(i32, i32)

; Unknown intrinsic
declare void @llvm.aie.put.wms(i32, i128)

; Unknown intrinsic
declare void @llvm.aie.put.fms(i32, float)

; Unknown intrinsic
declare i32 @llvm.aie.get.ss(i32)

; Unknown intrinsic
declare i128 @llvm.aie.get.wss(i32)

; Unknown intrinsic
declare float @llvm.aie.get.fss(i32)

; Unknown intrinsic
declare void @llvm.aie.put.mcd(i384)

; Unknown intrinsic
declare i384 @llvm.aie.get.scd()

; Unknown intrinsic
declare void @llvm.aie.lock.acquire.reg(i32, i32)

; Unknown intrinsic
declare void @llvm.aie.lock.release.reg(i32, i32)

define void @core_1_4() {
  store i32 14, ptr getelementptr inbounds (i8, ptr @a14, i64 12), align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
