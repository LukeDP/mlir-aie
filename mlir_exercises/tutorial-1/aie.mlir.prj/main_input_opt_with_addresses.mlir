module @tutorial_1 attributes {llvm.target_triple = "aie"} {
  llvm.mlir.global external @a14() {addr_space = 0 : i32} : !llvm.array<256 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.event0() attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.event1() attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.put.wms(i32, i128) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.put.fms(i32, f32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.get.ss(i32) -> i32 attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.get.wss(i32) -> i128 attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.get.fss(i32) -> f32 attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.put.mcd(i384) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.get.scd() -> i384 attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.lock.acquire.reg(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie.lock.release.reg(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @core_1_4() {
    %0 = llvm.mlir.addressof @a14 : !llvm.ptr
    %1 = llvm.mlir.constant(14 : i32) : i32
    %2 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<256 x i32>
    %3 = llvm.getelementptr inbounds|nuw %2[3] : (!llvm.ptr) -> !llvm.ptr, i32
    llvm.store %1, %3 : i32, !llvm.ptr
    llvm.return
  }
}

