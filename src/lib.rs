//  .--,       .--,
// ( (  \.---./  ) )
//  '.__/o   o\__.'
//     {=  ^  =}
//      >  -  <
//     /       \   JoJo
//    //       \\
//   //|   .   |\\
//   "'\       /'"_.-~^`'-.
//      \  _  /--'         `
//    ___)( )(___
//   (((__) (__)))
//
//! Leetcode Solutions.
//!
//! A rust implementation.
#![feature(test)]
extern crate test;

pub struct Solution;

/// # Q1. Dive Borad
///
/// 你正在使用一堆木板建造跳水板。有两种类型的木板，其中长度较短的木板长度为shorter，长度较长的木板长度为longer。
/// 你必须正好使用k块木板。编写一个方法，生成跳水板所有可能的长度。
///
/// 要求: 返回的长度需要从小到大排列。
///
/// # 示例：
///
///   输入：
///   ```shell
///   shorter = 1
///   longer = 2
///   k = 3
///   ```
///   输出： {3,4,5,6}
///   提示：
///       0 < shorter <= longer
///           0 <= k <= 100000
impl Solution {
    /// # Example
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::dive_board(1, 2, 3), vec![3, 4, 5, 6]);
    /// assert_eq!(Solution::dive_board(1, 1, 0), vec![]);
    /// assert_eq!(Solution::dive_board(1, 1, 100), vec![100]);
    /// ```
    ///
    pub fn dive_board(shorter: i32, longer: i32, k: i32) -> Vec<i32> {
        if k <= 0 {
            vec![]
        } else if shorter == longer {
            vec![shorter * k]
        } else {
            (0..=k)
                .map(|v| shorter * k + (longer - shorter) * v)
                .collect()
        }
    }
}

/// # Q2. Two Sum
///
/// 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那两个
/// 整数，并返回他们的数组下标。
///
/// 你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
///
/// 给定 nums = [2, 7, 11, 15], target = 9
/// 因为 nums[0] + nums[1] = 2 + 7 = 9
/// 所以返回 [0, 1]
impl Solution {
    /// # Example
    ///
    /// > time effecient
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::two_sum(vec![2, 7, 11, 5], 9), vec![0, 1]);
    /// assert_eq!(Solution::two_sum(vec![3, 2, 4], 6), vec![1, 2]);
    /// assert_eq!(Solution::two_sum(vec![4, 4, 5, 3], 8), vec![0, 1]);
    /// ```
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        use std::collections::HashMap;

        let mut map: HashMap<i32, usize> = HashMap::new();
        for (i, j) in nums.iter().enumerate() {
            if let Some(v) = map.get(&(target - j)) {
                return vec![*v as i32, i as i32];
            };
            map.insert(nums[i], i);
        }
        panic!()
    }

    /// # Example
    ///
    /// > memory effecient
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::two_sum2(vec![2, 7, 11, 5], 9), vec![0, 1]);
    /// assert_eq!(Solution::two_sum2(vec![3, 2, 4], 6), vec![1, 2]);
    /// assert_eq!(Solution::two_sum2(vec![4, 4, 5, 3], 8), vec![0, 1]);
    /// ```
    pub fn two_sum2(nums: Vec<i32>, target: i32) -> Vec<i32> {
        for i in 0..nums.len() {
            for j in (i + 1)..nums.len() {
                if nums[i] + nums[j] == target {
                    return vec![i as i32, j as i32];
                }
            }
        }
        panic!()
    }
}

/// # Q3. Find Median Sorted Arrays
///
/// 给定两个大小为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。
/// 请你找出这两个正序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
/// 你可以假设 nums1 和 nums2 不会同时为空。
///
/// # Example
///
/// 示例 1:
/// ```shell
/// nums1 = [1, 3]
/// nums2 = [2]
///
/// 则中位数是 2.0
/// ```
///
/// 示例 2:
/// ```shell
/// nums1 = [1, 2]
/// nums2 = [3, 4]
///
/// 则中位数是 (2 + 3)/2 = 2.5
/// ```
impl Solution {
    /// # Example
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::find_median_sorted_arrays(&[1, 3], &[2]), 2.0f64);
    /// assert_eq!(Solution::find_median_sorted_arrays(&[1, 2], &[3, 4]), 2.5f64);
    /// assert_eq!(Solution::find_median_sorted_arrays(&[1, 3, 5, 7, 8], &[4, 5, 9]), 5.0f64);
    /// assert_eq!(Solution::find_median_sorted_arrays(&[1, 3, 5, 7, 8], &[4]), 4.5f64);
    /// assert_eq!(Solution::find_median_sorted_arrays(&[1, 3, 5, 7, 8], &[]), 5.0f64);
    /// assert_eq!(Solution::find_median_sorted_arrays(&[1], &[4]), 2.5f64);
    /// assert_eq!(Solution::find_median_sorted_arrays(&[1], &[]), 1.0f64);
    /// assert_eq!(Solution::find_median_sorted_arrays(&[1, 2], &[-1, 3]), 1.5f64);
    /// ```
    ///
    /// # Benchmark
    ///
    /// Leetcode benchmark
    ///
    /// - time: 4ms
    /// - memory: 2m
    ///
    pub fn find_median_sorted_arrays(nums1: &[i32], nums2: &[i32]) -> f64 {
        use std::cmp;
        use std::i32;

        let mut index1: usize = 0;
        let mut index2: usize = 0;
        let target: usize = (nums1.len() + nums2.len()) / 2 + (nums1.len() + nums2.len()) % 2;
        while target > index1 + index2 {
            let remain = (target - index1 - index2) % 2;
            let lift = (target - index1 - index2) / 2;
            let (lift1, lift2) = if lift + remain > nums1.len() - index1 {
                (nums1.len() - index1, lift + index1 + remain - nums1.len())
            } else if lift + remain > nums2.len() - index2 {
                (lift + index2 + remain - nums2.len(), nums2.len() - index2)
            } else {
                (
                    cmp::min(lift + remain, nums1.len() - index1),
                    cmp::min(lift + remain, nums2.len() - index2),
                )
            };
            // let lift1: usize = cmp::min(cmp::max(left / 2, 1), nums1.len() - index1);
            // let lift2: usize = cmp::min(cmp::max(left - lift1, 1), nums2.len() - index2);
            if lift1 == 0 {
                index2 += lift2;
            } else if lift2 == 0 || nums1[index1 + lift1 - 1] <= nums2[index2 + lift2 - 1] {
                index1 += lift1;
            } else {
                index2 += lift2;
            };
        }

        if index1 == 0 && (nums1.len() + nums2.len()) % 2 == 0 {
            (nums2[index2 - 1]
                + cmp::min(
                    nums1.get(index1).unwrap_or(&i32::MAX),
                    nums2.get(index2).unwrap_or(&i32::MAX),
                )) as f64
                / 2f64
        } else if index2 == 0 && (nums1.len() + nums2.len()) % 2 == 0 {
            (nums1[index1 - 1]
                + cmp::min(
                    nums1.get(index1).unwrap_or(&i32::MAX),
                    nums2.get(index2).unwrap_or(&i32::MAX),
                )) as f64
                / 2f64
        } else if (nums1.len() + nums2.len()) % 2 == 0 {
            (cmp::max(nums1[index1 - 1], nums2[index2 - 1])
                + cmp::min(
                    nums1.get(index1).unwrap_or(&i32::MAX),
                    nums2.get(index2).unwrap_or(&i32::MAX),
                )) as f64
                / 2f64
        } else if index1 == 0 {
            nums2[index2 - 1] as f64
        } else if index2 == 0 {
            nums1[index1 - 1] as f64
        } else {
            cmp::max(nums1[index1 - 1], nums2[index2 - 1]) as f64
        }
    }
}

/// # Q300. Longest Increasing Subsequence
///
/// 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
/// [Leetcode300](https://leetcode-cn.com/problems/longest-increasing-subsequence/)
impl Solution {
    /// # Example
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::longest_increasing_subsequence(vec![10, 9, 2, 5, 3, 7, 101, 18]), (vec![2, 3, 7, 18], 4));
    /// assert_eq!(Solution::longest_increasing_subsequence(vec![0, 1, 0, 3, 2, 3]), (vec![0, 1, 2, 3], 4));
    /// assert_eq!(Solution::longest_increasing_subsequence(vec![7, 7, 7, 7, 7, 7, 7]), (vec![7], 1));
    /// ```
    pub fn longest_increasing_subsequence(nums: Vec<i32>) -> (Vec<i32>, i32) {
        let mut lis = vec![1i32; nums.len()];
        for i in 1..nums.len() {
            for j in 0..i {
                if nums[i] > nums[j] && lis[i] < lis[j] + 1 {
                    lis[i] = lis[j] + 1
                }
            }
        }

        // length of lis
        let len_lis = *lis.iter().max().unwrap();
        let mut stepper = len_lis;

        // longest increasing subsequence
        let lis: Vec<i32> = lis
            .into_iter()
            .enumerate()
            .rev()
            .filter_map(|(i, v)| {
                (stepper == v).then(|| {
                    stepper -= 1;
                    nums[i]
                })
            })
            // .rev() // #FIXME, double rev give `filter_map` a original forward?
            .collect();
        (lis.into_iter().rev().collect(), len_lis)
    }
}

/// # Q1143. Longest Common Subsequence
///
/// [Leetcode1143](https://leetcode.com/problems/longest-common-subsequence/)
///
/// 给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。
///
/// 一个字符串的 子序列
/// 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
/// 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde"
/// 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。
///
/// 若这两个字符串没有公共子序列，则返回 0。
///
/// 示例 1:
///
/// 输入：text1 = "abcde", text2 = "ace"
/// 输出：3
/// 解释：最长公共子序列是 "ace"，它的长度为 3。
///
/// 示例 2:
///
/// 输入：text1 = "abc", text2 = "abc"
/// 输出：3
/// 解释：最长公共子序列是 "abc"，它的长度为 3。
///
/// 示例 3:
///
/// 输入：text1 = "abc", text2 = "def"
/// 输出：0
/// 解释：两个字符串没有公共子序列，返回 0。
///
/// 提示:
///     1 <= text1.length <= 1000
///     1 <= text2.length <= 1000
///     输入的字符串只含有小写英文字符。
impl Solution {
    /// # Solution
    ///
    ///   a b c d e
    /// a 1 1 1 1 1
    /// c 1 1 2 2 2
    /// e 1 1 2 2 3
    ///
    /// # Example
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::longest_common_subsequence(String::from("abcde"), String::from("ace")), 3);
    /// ```
    ///
    /// # Benchmark
    ///
    /// time ~ O(m*n)
    ///
    pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
        let (slice1, slice2) = (text1.as_bytes(), text2.as_bytes());
        let (m, n) = (text1.len(), text2.len());
        let mut cumu = vec![vec![0i32; n + 1]; m + 1];
        for i in 1..(m + 1) {
            for j in 1..(n + 1) {
                if slice1[i - 1] == slice2[j - 1] {
                    cumu[i][j] = cumu[i - 1][j - 1] + 1;
                } else {
                    cumu[i][j] = i32::max(cumu[i][j - 1], cumu[i - 1][j]);
                }
            }
        }
        cumu[m][n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_find_median_sorted_arrays(b: &mut Bencher) {
        b.iter(|| {
            Solution::find_median_sorted_arrays(&[1, 3], &[2]);
            Solution::find_median_sorted_arrays(&[1, 2], &[3, 4]);
            Solution::find_median_sorted_arrays(&[1, 3, 5, 7, 8], &[4, 5, 9]);
            Solution::find_median_sorted_arrays(&[1, 3, 5, 7, 8], &[4]);
            Solution::find_median_sorted_arrays(&[1, 3, 5, 7, 8], &[]);
            Solution::find_median_sorted_arrays(&[1], &[4]);
            Solution::find_median_sorted_arrays(&[1], &[]);
            Solution::find_median_sorted_arrays(&[1, 2], &[-1, 3]);
        });
    }
}
