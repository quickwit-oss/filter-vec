use std::arch::x86_64::__m512i as DataType;
use std::arch::x86_64::_mm512_add_epi32 as op_add;
use std::arch::x86_64::_mm512_cmple_epi32_mask as op_less_or_equal;
use std::arch::x86_64::_mm512_loadu_epi32 as load_unaligned;
use std::arch::x86_64::_mm512_set1_epi32 as set1;
use std::arch::x86_64::*;
use std::ops::RangeInclusive;

const NUM_LANES: usize = 16;

pub fn filter_vec(input: &[u32], range: RangeInclusive<u32>, output: &mut Vec<u32>) {
    assert_eq!(input.len() % NUM_LANES, 0);
    // We restrict the accepted bondary, because unsigned integers & SIMD don't
    // play well.
    let accepted_range = 0u32..(i32::MAX as u32);
    assert!(accepted_range.contains(range.start()));
    assert!(accepted_range.contains(range.end()));
    output.clear();
    output.reserve(input.len());
    let num_words = input.len() / NUM_LANES;
    unsafe {
        let output_len = filter_vec_aux(
            input.as_ptr(),
            range,
            output.as_mut_ptr(),
            num_words,
        );
        output.set_len(output_len);
    }
}


pub unsafe fn filter_vec_aux(
    mut input: *const u32,
    range: RangeInclusive<u32>,
    output: *mut u32,
    num_words: usize,
) -> usize {
    let mut output_len = 0;
    let mut output_end = output;
    let range_simd = set1(*range.start() as i32)..=set1(*range.end() as i32);
    let mut docs = from_u32x16([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    const SHIFT: __m512i = from_u32x16([NUM_LANES as u32; NUM_LANES]);
    for _ in 0..num_words {
        let word = load_unaligned(input as *const i32);
        let mask = is_between(word, range_simd.clone());
        _mm512_mask_compressstoreu_epi32(output_end as *mut u8, mask, docs);
        let added_len = mask.count_ones();
        output_end = output_end.offset(added_len as isize);
        docs = op_add(docs, SHIFT);
        input = input.offset(1);
    }
    output_end.offset_from(output) as usize
}

#[inline]
unsafe fn is_between(val: DataType, range: std::ops::RangeInclusive<DataType>) -> u16 {
    let low = op_less_or_equal(*range.start(), val);
    let high = op_less_or_equal(val, *range.end());
    low & high
}

union U8x64 {
    vector: DataType,
    vals: [u32; NUM_LANES],
}

const fn from_u32x16(vals: [u32; NUM_LANES]) -> DataType {
    unsafe { U8x64 { vals }.vector }
}

const fn from_data(vector: DataType) -> [u32; NUM_LANES] {
    unsafe { U8x64 { vector }.vals }
}
