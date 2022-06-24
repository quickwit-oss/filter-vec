RUSTFLAGS='-C target-cpu=skylake-avx512' cargo +nightly asm filter_vec::avx512::filter_vec_aux


AGPL:
filter-interval/avx2    time:   [285.77 us 286.53 us 287.29 us]
                        thrpt:  [3.6498 Gelem/s 3.6596 Gelem/s 3.6693 Gelem/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
filter-interval/avx512  time:   [121.96 us 122.05 us 122.15 us]
                        thrpt:  [8.5840 Gelem/s 8.5917 Gelem/s 8.5977 Gelem/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
filter-interval/scalar_iterator
                        time:   [6.0165 ms 6.0311 ms 6.0461 ms]
                        thrpt:  [173.43 Melem/s 173.86 Melem/s 174.28 Melem/s]
Found 6 outliers among 100 measurements (6.00%)
  1 (1.00%) low mild
  5 (5.00%) high mild
filter-interval/scalar_forloop
                        time:   [5.0963 ms 5.1004 ms 5.1045 ms]
                        thrpt:  [205.42 Melem/s 205.59 Melem/s 205.75 Melem/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
filter-interval/scalar_nobranch
                        time:   [3.4513 ms 3.4548 ms 3.4585 ms]
                        thrpt:  [303.18 Melem/s 303.51 Melem/s 303.82 Melem/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high mild
