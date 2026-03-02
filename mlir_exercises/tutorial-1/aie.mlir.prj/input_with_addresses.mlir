module @tutorial_1 {
  aie.device(xcvc1902) {
    %tile_1_4 = aie.tile(1, 4)
    %a14 = aie.buffer(%tile_1_4) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "a14"} : memref<256xi32> 
    %core_1_4 = aie.core(%tile_1_4) {
      %c14_i32 = arith.constant 14 : i32
      %c3 = arith.constant 3 : index
      memref.store %c14_i32, %a14[%c3] : memref<256xi32>
      aie.end
    }
  }
}
