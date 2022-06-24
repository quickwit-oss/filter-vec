#![feature(stdsimd)]
/*
Benchmark results
====================================================

filter-interval/scalar_iterator:
    thrpt:  [271.47 Melem/s 272.77 Melem/s 273.97 Melem/s]
filter-interval/scalar_forloop:
    thrpt:  [300.44 Melem/s 300.92 Melem/s 301.38 Melem/s]
filter-interval/scalar_nobranch:
    thrpt:  [370.27 Melem/s 372.01 Melem/s 373.60 Melem/s]
filter-interval/avx2
    thrpt:  [2.2039 Gelem/s 2.2491 Gelem/s 2.2940 Gelem/s]
*/

pub mod avx2;
pub mod avx512;

use std::ops::RangeInclusive;

// -------------------------------------------------------------------------------------------
// Scalar version with a for-loop

pub fn filter_vec_scalar(input: &[u32], range: RangeInclusive<u32>, output: &mut Vec<u32>) {
    output.clear();
    output.reserve(input.len());
    for (id, &el) in input.iter().enumerate() {
        if range.contains(&el) {
            output.push(id as u32);
        }
    }
}

// -------------------------------------------------------------------------------------------
// Branchless Scalar version

pub fn filter_vec_nobranch(input: &[u32], range: RangeInclusive<u32>, output: &mut Vec<u32>) {
    output.clear();
    output.resize(input.len(), 0u32);
    let mut output_len = 0;
    for (id, &el) in input.iter().enumerate() {
        output[output_len] = id as u32;
        output_len += if range.contains(&el) { 1 } else { 0 };
    }
    output.truncate(output_len);
}

// -------------------------------------------------------------------------------------------
// Iterator version

pub fn filter_vec_iter(input: &[u32], range: RangeInclusive<u32>, output: &mut Vec<u32>) {
    output.clear();
    output.reserve(input.len());
    output.extend(
        input
            .iter()
            .cloned()
            .enumerate()
            .filter(|&(_, el)| range.contains(&el))
            .map(|(id, _)| id as u32),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter() {
        let v = &[14, 3, 15, 3, 5, 14, 2, 3, 2, 3, 3, 4, 6, 10, 3, 7];
        let interval = 3..=5;
        let expected: Vec<u32> = v
            .iter()
            .cloned()
            .enumerate()
            .filter(|&(_, el)| interval.contains(&el))
            .map(|(ord, _)| ord as u32)
            .collect();
        {
            let mut output = Vec::new();
            super::avx2::filter_vec(&v[..], interval.clone(), &mut output);
            assert_eq!(&output[..], &expected);
        }
        {
            let mut output = Vec::new();
            filter_vec_nobranch(&v[..], interval.clone(), &mut output);
            assert_eq!(&output[..], &expected);
        }
        {
            let mut output = Vec::new();
            filter_vec_iter(&v[..], interval.clone(), &mut output);
            assert_eq!(&output[..], &expected);
        }
       {
            let mut output = Vec::new();
            super::avx512::filter_vec(&v[..], interval.clone(), &mut output);
            assert_eq!(&output[..], &expected);
        }
    }
}
