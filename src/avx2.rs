use std::arch::x86_64::__m256i as DataType;
use std::arch::x86_64::_mm256_add_epi32 as op_add;
use std::arch::x86_64::_mm256_cmpgt_epi32 as op_greater;
use std::arch::x86_64::_mm256_lddqu_si256 as load_unaligned;
use std::arch::x86_64::_mm256_or_si256 as op_or;
use std::arch::x86_64::_mm256_set1_epi32 as set1;
use std::arch::x86_64::_mm256_storeu_si256 as store_unaligned;
use std::arch::x86_64::*;
use std::ops::RangeInclusive;

const NUM_LANES: usize = 8;

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
        let output_len = filter_vec_avx2_aux(
            input.as_ptr() as *const __m256i,
            range,
            output.as_mut_ptr(),
            num_words,
        );
        output.set_len(output_len);
    }
}

unsafe fn filter_vec_avx2_aux(
    mut input: *const __m256i,
    range: RangeInclusive<u32>,
    output: *mut u32,
    num_words: usize,
) -> usize {
    let mut output_tail = output;
    let range_simd = set1(*range.start() as i32)..=set1(*range.end() as i32);
    let mut ids = from_u32x8([0, 1, 2, 3, 4, 5, 6, 7]);
    const SHIFT: __m256i = from_u32x8([NUM_LANES as u32; NUM_LANES]);
    for _ in 0..num_words {
        let word = load_unaligned(input);
        let keeper_bitset = compute_filter_bitset(word, range_simd.clone());
        let added_len = keeper_bitset.count_ones();
        let filtered_doc_ids = compact(ids, keeper_bitset);
        store_unaligned(
            output_tail as *mut __m256i,
            filtered_doc_ids,
        );
        output_tail = output_tail.offset(added_len as isize);
        ids = op_add(ids, SHIFT);
        input = input.offset(1);
    }
    output_tail.offset_from(output) as usize
}

#[inline]
unsafe fn compact(data: DataType, mask: u8) -> DataType {
    let vperm_mask = MASK_TO_PERMUTATION[mask as usize];
    _mm256_permutevar8x32_epi32(data, vperm_mask)
}

#[inline]
unsafe fn compute_filter_bitset(val: __m256i, range: std::ops::RangeInclusive<__m256i>) -> u8 {
    let too_low = op_greater(*range.start(), val);
    let too_high = op_greater(val,*range.end());
    let inside = op_or(too_low, too_high);
    255 - std::arch::x86_64::_mm256_movemask_ps(std::mem::transmute::<DataType, __m256>(inside))
        as u8
}

union U8x32 {
    vector: DataType,
    vals: [u32; NUM_LANES],
}

const fn from_u32x8(vals: [u32; NUM_LANES]) -> DataType {
    unsafe { U8x32 { vals }.vector }
}

const MASK_TO_PERMUTATION: [DataType; 256] = [
    from_u32x8([0, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 0, 0, 0, 0, 0, 0]),
    from_u32x8([2, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 0, 0, 0, 0, 0]),
    from_u32x8([3, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 3, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 0, 0, 0, 0, 0]),
    from_u32x8([2, 3, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 3, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 0, 0, 0, 0]),
    from_u32x8([4, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 4, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 0, 0, 0, 0, 0]),
    from_u32x8([2, 4, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 4, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 0, 0, 0, 0]),
    from_u32x8([3, 4, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 0, 0, 0, 0, 0]),
    from_u32x8([1, 3, 4, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 0, 0, 0, 0]),
    from_u32x8([2, 3, 4, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 0, 0, 0, 0]),
    from_u32x8([1, 2, 3, 4, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 0, 0, 0]),
    from_u32x8([5, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 5, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 5, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 5, 0, 0, 0, 0, 0]),
    from_u32x8([2, 5, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 5, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 5, 0, 0, 0, 0]),
    from_u32x8([3, 5, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 5, 0, 0, 0, 0, 0]),
    from_u32x8([1, 3, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 5, 0, 0, 0, 0]),
    from_u32x8([2, 3, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 5, 0, 0, 0, 0]),
    from_u32x8([1, 2, 3, 5, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 5, 0, 0, 0]),
    from_u32x8([4, 5, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 5, 0, 0, 0, 0, 0]),
    from_u32x8([1, 4, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 5, 0, 0, 0, 0]),
    from_u32x8([2, 4, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 5, 0, 0, 0, 0]),
    from_u32x8([1, 2, 4, 5, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 5, 0, 0, 0]),
    from_u32x8([3, 4, 5, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 5, 0, 0, 0, 0]),
    from_u32x8([1, 3, 4, 5, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 5, 0, 0, 0]),
    from_u32x8([2, 3, 4, 5, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 5, 0, 0, 0]),
    from_u32x8([1, 2, 3, 4, 5, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 5, 0, 0]),
    from_u32x8([6, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 6, 0, 0, 0, 0, 0]),
    from_u32x8([2, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 6, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 6, 0, 0, 0, 0]),
    from_u32x8([3, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 6, 0, 0, 0, 0, 0]),
    from_u32x8([1, 3, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 6, 0, 0, 0, 0]),
    from_u32x8([2, 3, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 6, 0, 0, 0, 0]),
    from_u32x8([1, 2, 3, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 6, 0, 0, 0]),
    from_u32x8([4, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 6, 0, 0, 0, 0, 0]),
    from_u32x8([1, 4, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 6, 0, 0, 0, 0]),
    from_u32x8([2, 4, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 6, 0, 0, 0, 0]),
    from_u32x8([1, 2, 4, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 6, 0, 0, 0]),
    from_u32x8([3, 4, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 6, 0, 0, 0, 0]),
    from_u32x8([1, 3, 4, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 6, 0, 0, 0]),
    from_u32x8([2, 3, 4, 6, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 6, 0, 0, 0]),
    from_u32x8([1, 2, 3, 4, 6, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 6, 0, 0]),
    from_u32x8([5, 6, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 5, 6, 0, 0, 0, 0, 0]),
    from_u32x8([1, 5, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 5, 6, 0, 0, 0, 0]),
    from_u32x8([2, 5, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 5, 6, 0, 0, 0, 0]),
    from_u32x8([1, 2, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 5, 6, 0, 0, 0]),
    from_u32x8([3, 5, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 5, 6, 0, 0, 0, 0]),
    from_u32x8([1, 3, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 5, 6, 0, 0, 0]),
    from_u32x8([2, 3, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 5, 6, 0, 0, 0]),
    from_u32x8([1, 2, 3, 5, 6, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 5, 6, 0, 0]),
    from_u32x8([4, 5, 6, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 5, 6, 0, 0, 0, 0]),
    from_u32x8([1, 4, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 5, 6, 0, 0, 0]),
    from_u32x8([2, 4, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 5, 6, 0, 0, 0]),
    from_u32x8([1, 2, 4, 5, 6, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 5, 6, 0, 0]),
    from_u32x8([3, 4, 5, 6, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 5, 6, 0, 0, 0]),
    from_u32x8([1, 3, 4, 5, 6, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 5, 6, 0, 0]),
    from_u32x8([2, 3, 4, 5, 6, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 5, 6, 0, 0]),
    from_u32x8([1, 2, 3, 4, 5, 6, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 5, 6, 0]),
    from_u32x8([7, 0, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([1, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 7, 0, 0, 0, 0, 0]),
    from_u32x8([2, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 7, 0, 0, 0, 0, 0]),
    from_u32x8([1, 2, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 7, 0, 0, 0, 0]),
    from_u32x8([3, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 7, 0, 0, 0, 0, 0]),
    from_u32x8([1, 3, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 7, 0, 0, 0, 0]),
    from_u32x8([2, 3, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 7, 0, 0, 0, 0]),
    from_u32x8([1, 2, 3, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 7, 0, 0, 0]),
    from_u32x8([4, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 7, 0, 0, 0, 0, 0]),
    from_u32x8([1, 4, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 7, 0, 0, 0, 0]),
    from_u32x8([2, 4, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 7, 0, 0, 0, 0]),
    from_u32x8([1, 2, 4, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 7, 0, 0, 0]),
    from_u32x8([3, 4, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 7, 0, 0, 0, 0]),
    from_u32x8([1, 3, 4, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 7, 0, 0, 0]),
    from_u32x8([2, 3, 4, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 7, 0, 0, 0]),
    from_u32x8([1, 2, 3, 4, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 7, 0, 0]),
    from_u32x8([5, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 5, 7, 0, 0, 0, 0, 0]),
    from_u32x8([1, 5, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 5, 7, 0, 0, 0, 0]),
    from_u32x8([2, 5, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 5, 7, 0, 0, 0, 0]),
    from_u32x8([1, 2, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 5, 7, 0, 0, 0]),
    from_u32x8([3, 5, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 5, 7, 0, 0, 0, 0]),
    from_u32x8([1, 3, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 5, 7, 0, 0, 0]),
    from_u32x8([2, 3, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 5, 7, 0, 0, 0]),
    from_u32x8([1, 2, 3, 5, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 5, 7, 0, 0]),
    from_u32x8([4, 5, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 5, 7, 0, 0, 0, 0]),
    from_u32x8([1, 4, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 5, 7, 0, 0, 0]),
    from_u32x8([2, 4, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 5, 7, 0, 0, 0]),
    from_u32x8([1, 2, 4, 5, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 5, 7, 0, 0]),
    from_u32x8([3, 4, 5, 7, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 5, 7, 0, 0, 0]),
    from_u32x8([1, 3, 4, 5, 7, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 5, 7, 0, 0]),
    from_u32x8([2, 3, 4, 5, 7, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 5, 7, 0, 0]),
    from_u32x8([1, 2, 3, 4, 5, 7, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 5, 7, 0]),
    from_u32x8([6, 7, 0, 0, 0, 0, 0, 0]),
    from_u32x8([0, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([1, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 1, 6, 7, 0, 0, 0, 0]),
    from_u32x8([2, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 2, 6, 7, 0, 0, 0, 0]),
    from_u32x8([1, 2, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 2, 6, 7, 0, 0, 0]),
    from_u32x8([3, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 3, 6, 7, 0, 0, 0, 0]),
    from_u32x8([1, 3, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 3, 6, 7, 0, 0, 0]),
    from_u32x8([2, 3, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 3, 6, 7, 0, 0, 0]),
    from_u32x8([1, 2, 3, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 3, 6, 7, 0, 0]),
    from_u32x8([4, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 4, 6, 7, 0, 0, 0, 0]),
    from_u32x8([1, 4, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 4, 6, 7, 0, 0, 0]),
    from_u32x8([2, 4, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 4, 6, 7, 0, 0, 0]),
    from_u32x8([1, 2, 4, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 4, 6, 7, 0, 0]),
    from_u32x8([3, 4, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 3, 4, 6, 7, 0, 0, 0]),
    from_u32x8([1, 3, 4, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 3, 4, 6, 7, 0, 0]),
    from_u32x8([2, 3, 4, 6, 7, 0, 0, 0]),
    from_u32x8([0, 2, 3, 4, 6, 7, 0, 0]),
    from_u32x8([1, 2, 3, 4, 6, 7, 0, 0]),
    from_u32x8([0, 1, 2, 3, 4, 6, 7, 0]),
    from_u32x8([5, 6, 7, 0, 0, 0, 0, 0]),
    from_u32x8([0, 5, 6, 7, 0, 0, 0, 0]),
    from_u32x8([1, 5, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 1, 5, 6, 7, 0, 0, 0]),
    from_u32x8([2, 5, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 2, 5, 6, 7, 0, 0, 0]),
    from_u32x8([1, 2, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 2, 5, 6, 7, 0, 0]),
    from_u32x8([3, 5, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 3, 5, 6, 7, 0, 0, 0]),
    from_u32x8([1, 3, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 3, 5, 6, 7, 0, 0]),
    from_u32x8([2, 3, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 2, 3, 5, 6, 7, 0, 0]),
    from_u32x8([1, 2, 3, 5, 6, 7, 0, 0]),
    from_u32x8([0, 1, 2, 3, 5, 6, 7, 0]),
    from_u32x8([4, 5, 6, 7, 0, 0, 0, 0]),
    from_u32x8([0, 4, 5, 6, 7, 0, 0, 0]),
    from_u32x8([1, 4, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 1, 4, 5, 6, 7, 0, 0]),
    from_u32x8([2, 4, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 2, 4, 5, 6, 7, 0, 0]),
    from_u32x8([1, 2, 4, 5, 6, 7, 0, 0]),
    from_u32x8([0, 1, 2, 4, 5, 6, 7, 0]),
    from_u32x8([3, 4, 5, 6, 7, 0, 0, 0]),
    from_u32x8([0, 3, 4, 5, 6, 7, 0, 0]),
    from_u32x8([1, 3, 4, 5, 6, 7, 0, 0]),
    from_u32x8([0, 1, 3, 4, 5, 6, 7, 0]),
    from_u32x8([2, 3, 4, 5, 6, 7, 0, 0]),
    from_u32x8([0, 2, 3, 4, 5, 6, 7, 0]),
    from_u32x8([1, 2, 3, 4, 5, 6, 7, 0]),
    from_u32x8([0, 1, 2, 3, 4, 5, 6, 7]),
];
