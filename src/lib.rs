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

use std::collections::BTreeMap;
use std::collections::HashMap;

pub struct Solution;

/// # 1. Dive Borad
///
/// 你正在使用一堆木板建造跳水板。有两种类型的木板，其中长度较短的木板长度为shorter，长度较长的木板长度为longer。
/// 你必须正好使用k块木板。编写一个方法，生成跳水板所有可能的长度。
///
/// 要求: 返回的长度需要从小到大排列。
///
/// 示例：
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
    /// # Benchmark
    ///
    /// Leetcode benchmark
    /// - time: 20ms
    /// - memory: 2.8MB
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

/// # 2. Two Sum
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
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        Solution::two_sum_hashmap(nums, target)
    }

    /// # Example
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::two_sum_hashmap(vec![2, 7, 11, 5], 9), vec![0, 1]);
    /// assert_eq!(Solution::two_sum_hashmap(vec![3, 2, 4], 6), vec![1, 2]);
    /// assert_eq!(Solution::two_sum_hashmap(vec![4, 4, 5, 3], 8), vec![0, 1]);
    /// ```
    ///
    /// # Benchmark
    ///
    /// Leetcode benchmark
    /// - time: 32ms
    /// - memory: 2MB
    pub fn two_sum_hashmap(nums: Vec<i32>, target: i32) -> Vec<i32> {
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
    /// assert_eq!(Solution::two_sum_traverse(vec![2, 7, 11, 5], 9), vec![0, 1]);
    /// assert_eq!(Solution::two_sum_traverse(vec![3, 2, 4], 6), vec![1, 2]);
    /// assert_eq!(Solution::two_sum_traverse(vec![4, 4, 5, 3], 8), vec![0, 1]);
    /// ```
    pub fn two_sum_traverse(nums: Vec<i32>, target: i32) -> Vec<i32> {
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

/// # 3. Find Median Sorted Arrays
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

/// # 153. 寻找旋转排序数组中的最小值
///
/// 已知一个长度为 n 的数组, 预先按照升序排列, 经由 1 到 n 次 旋转 后, 得到输入数组。例如, 原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
/// 若旋转 4 次, 则可以得到 [4,5,6,7,0,1,2]
/// 若旋转 7 次, 则可以得到 [0,1,2,4,5,6,7]
///
/// 注意, 数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。
///
/// 给你一个元素值 互不相同 的数组 nums , 它原来是一个升序排列的数组, 并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
///
///
/// 示例 1：
///
/// 输入：nums = [3,4,5,1,2]
/// 输出：1
/// 解释：原数组为 [1,2,3,4,5] , 旋转 3 次得到输入数组。
///
/// 示例 2：
///
/// 输入：nums = [4,5,6,7,0,1,2]
/// 输出：0
/// 解释：原数组为 [0,1,2,4,5,6,7] , 旋转 4 次得到输入数组。
///
/// 示例 3：
///
/// 输入：nums = [11,13,15,17]
/// 输出：11
/// 解释：原数组为 [11,13,15,17] , 旋转 4 次得到输入数组。
///
///
/// 提示：
///     n == nums.length
///     1 <= n <= 5000
///     -5000 <= nums[i] <= 5000
///     nums 中的所有整数 互不相同
///     nums 原来是一个升序排序的数组, 并进行了 1 至 n 次旋转
impl Solution {
    /// # Example
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::find_min_in_rotated_sorted_arrary(vec![4,5,6,7,0,1,2]), 0);
    /// assert_eq!(Solution::find_min_in_rotated_sorted_arrary(vec![3,4,5,1,2]), 1);
    /// assert_eq!(Solution::find_min_in_rotated_sorted_arrary(vec![11,13,15,17]), 11);
    /// ```
    ///
    /// # Benchmark
    ///
    /// Leetcode benchmark
    /// - time 0ms
    /// - memory 1.9m
    pub fn find_min_in_rotated_sorted_arrary(nums: Vec<i32>) -> i32 {
        // Commented for memory saving.
        // assert!(nums.len() >= 1);
        let mut low: usize = 0;
        let mut high: usize = nums.len() - 1;
        // if not rotated or len == 1
        if nums[low] <= nums[high] {
            return nums[low];
        };
        while low + 1 < high {
            let pivot: usize = (low + high) / 2;
            if nums[low] < nums[pivot] {
                low = pivot;
            } else {
                high = pivot;
            };
            // Commented for memory saving.
            // } else if nums[high] > nums[pivot] {
            //     high = pivot;
            // } else {
            //     unreachable!();
            // };
        }
        i32::min(nums[low], nums[high])
    }
}

/// # 287 寻找重复数
///
/// [Leetcode287](https://leetcode-cn.com/problems/find-the-duplicate-number)
///
/// 给定一个包含 n + 1 个整数的数组 nums ，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。
/// 假设 nums 只有 一个重复的整数 ，找出 这个重复的数 。
///
/// 示例 1：
/// 输入：nums = [1,3,4,2,2]
/// 输出：2
///
/// 示例 2：
/// 输入：nums = [3,1,3,4,2]
/// 输出：3
///
/// 示例 3：
/// 输入：nums = [1,1]
/// 输出：1
///
/// 示例 4：
/// 输入：nums = [1,1,2]
/// 输出：1
///
/// 提示：
///     2 <= n <= 3 * 104
///     nums.length == n + 1
///     1 <= nums[i] <= n
///     nums 中 只有一个整数 出现 两次或多次 ，其余整数均只出现 一次
///
///
/// 进阶：
///     如何证明 nums 中至少存在一个重复的数字?
///     你可以在不修改数组 nums 的情况下解决这个问题吗？
///     你可以只用常量级 O(1) 的额外空间解决这个问题吗？
///     你可以设计一个时间复杂度小于 O(n2) 的解决方案吗？
impl Solution {
    pub fn find_the_duplicate_number(nums: Vec<i32>) -> i32 {
        Solution::find_the_duplicate_number_hashmap(nums)
    }

    /// # Examples
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::find_the_duplicate_number_btreemap(vec![1,3,4,2,2]), 2);
    /// assert_eq!(Solution::find_the_duplicate_number_btreemap(vec![3,1,3,4,2]), 3);
    /// assert_eq!(Solution::find_the_duplicate_number_btreemap(vec![1,1]), 1);
    /// assert_eq!(Solution::find_the_duplicate_number_btreemap(vec![1,1,2]), 1);
    /// ```
    ///
    /// # Benchmark
    ///
    /// Leetcode benchmark
    /// - time: 4ms, 5.56%
    /// - memory: 2.3MB, 50.00%
    pub fn find_the_duplicate_number_btreemap(nums: Vec<i32>) -> i32 {
        let mut container: BTreeMap<i32, bool> = BTreeMap::new();
        for i in nums.into_iter() {
            if let Some(_) = container.get(&i) {
                return i;
            } else {
                container.insert(i, true);
            }
        }
        panic!()
    }

    /// # Examples
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::find_the_duplicate_number_hashmap(vec![1,3,4,2,2]), 2);
    /// assert_eq!(Solution::find_the_duplicate_number_hashmap(vec![3,1,3,4,2]), 3);
    /// assert_eq!(Solution::find_the_duplicate_number_hashmap(vec![1,1]), 1);
    /// assert_eq!(Solution::find_the_duplicate_number_hashmap(vec![1,1,2]), 1);
    /// ```
    ///
    /// # Benchmark
    ///
    /// Leetcode benchmark
    /// - time: 0ms, 100%
    /// - memory: 2.5MB, 5.55%
    pub fn find_the_duplicate_number_hashmap(nums: Vec<i32>) -> i32 {
        let mut container: HashMap<i32, bool> = HashMap::new();
        for i in nums.into_iter() {
            if let Some(_) = container.get(&i) {
                return i;
            } else {
                container.insert(i, true);
            }
        }
        panic!()
    }
}

/// # 300. 最长递增子序列
///
/// [Leetcode300](https://leetcode-cn.com/problems/longest-increasing-subsequence/)
///
/// 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
/// 子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
///
/// 示例 1：
/// 输入：nums = [10,9,2,5,3,7,101,18]
/// 输出：4
/// 解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
///
/// 示例 2：
/// 输入：nums = [0,1,0,3,2,3]
/// 输出：4
///
/// 示例 3：
/// 输入：nums = [7,7,7,7,7,7,7]
/// 输出：1
///
///
/// 提示：
///     1 <= nums.length <= 2500
///     -104 <= nums[i] <= 104
///
/// 进阶：
///     你可以设计时间复杂度为 O(n2) 的解决方案吗？
///     你能将算法的时间复杂度降低到 O(n log(n)) 吗?
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
    ///
    /// # Benchmark
    ///
    /// Leetcode benchmark
    /// - time: 72ms
    /// - memory: 2MB
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

/// # 1143. 最长公共子序列
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

/// # LCP 37. 最小矩形面积
///
/// 二维平面上有 NNN 条直线，形式为 y = kx + b，其中 k、b为整数 且 k > 0。所有直线以 [k,b] 的形式存于二维数组 lines 中，不存在重合的两条直线。两两直线之间可能存在一个交点，最多会有 C(subN)(sup2)个交点。我们用一个平行于坐标轴的矩形覆盖所有的交点，请问这个矩形最小面积是多少。若直线之间无交点、仅有一个交点或所有交点均在同一条平行坐标轴的直线上，则返回0。
///
/// 注意：返回结果是浮点数，与标准答案 绝对误差或相对误差 在 10^-4 以内的结果都被视为正确结果
///
/// 示例 1：
///     输入：lines = [[2,3],[3,0],[4,1]]
///     输出：48.00000
///     解释：三条直线的三个交点为 (3, 9) (1, 5) 和 (-1, -3)。最小覆盖矩形左下角为 (-1, -3) 右上角为 (3,9)，面积为 48
///
/// 示例 2：
///     输入：lines = [[1,1],[2,3]]
///     输出：0.00000
///     解释：仅有一个交点 (-2，-1）
///
/// 限制：
///     1 <= lines.length <= 10^5 且 lines[i].length == 2
///     1 <= lines[0] <= 10000
///     -10000 <= lines[1] <= 10000
///     与标准答案绝对误差或相对误差在 10^-4 以内的结果都被视为正确结果
impl Solution {
    pub fn min_rec_size(lines: Vec<Vec<i32>>) -> f64 {
        Solution::min_rec_size_inplace_sort(lines)
    }

    /// # Examples
    ///
    /// ```rust
    /// use leetcode::Solution;
    ///
    /// assert_eq!(Solution::min_rec_size_inplace_sort(vec![vec![1,1],vec![2,3]]), 0.0f64);
    /// assert_eq!(Solution::min_rec_size_inplace_sort(vec![vec![2,3],vec![3,0],vec![4,1]]), 48.0f64);
    /// assert_eq!(Solution::min_rec_size_inplace_sort(vec![vec![2,0],vec![4,-3],vec![2,4],vec![1,-2],vec![1,-1]]), 180.5f64);
    /// assert_eq!(Solution::min_rec_size_inplace_sort(vec![vec![5,3],vec![3,-3],vec![5,1],vec![5,-5],vec![5,-2]]), 48.0f64);
    /// assert_eq!(Solution::min_rec_size_inplace_sort(vec![vec![4,2],vec![5,-1],vec![4,-4],vec![4,0],vec![5,-5]]), 460.0f64);
    /// ```
    ///
    /// # Benchmark
    ///
    /// Leetcode benchmark
    /// - time: 36ms
    /// - memory: 8.4MB
    pub fn min_rec_size_inplace_sort(mut lines: Vec<Vec<i32>>) -> f64 {
        if lines.len() <= 2 {
            return 0f64;
        }
        // sort by k
        lines.sort_unstable_by(|a, b| {
            a[0].partial_cmp(&b[0])
                .unwrap()
                .then(a[1].partial_cmp(&b[1]).unwrap())
        });
        let mut x_min = f64::MAX;
        let mut x_max = f64::MIN;
        let mut y_min = f64::MAX;
        let mut y_max = f64::MIN;
        let mut iter = lines.iter();
        let mut prev_up = iter.next().unwrap();
        let mut prev_down = prev_up;
        let mut curr_up = loop {
            let v = if let Some(v) = iter.next() {
                v
            } else {
                return 0f64;
            };
            if v[0] == prev_up[0] {
                prev_up = v;
                continue;
            };
            break v;
        };
        let mut curr_down = curr_up;
        let mut next = iter.next();
        loop {
            if next.is_some() && curr_up[0] == next.unwrap()[0] {
                curr_up = next.unwrap();
                next = iter.next();
                continue;
            };
            // min
            if curr_up[0] == prev_down[0] {
            } else {
                let x_cross = (f64::from(prev_down[1]) - f64::from(curr_up[1]))
                    / (f64::from(curr_up[0]) - f64::from(prev_down[0]));
                let y_cross = (f64::from(curr_up[0]) * f64::from(prev_down[1])
                    - f64::from(prev_down[0]) * f64::from(curr_up[1]))
                    / (f64::from(curr_up[0]) - f64::from(prev_down[0]));
                if x_cross < x_min {
                    x_min = x_cross;
                };
                if y_cross < y_min {
                    y_min = y_cross;
                };
                prev_down = curr_down;
            };
            // max
            if curr_down[0] == prev_up[0] {
            } else {
                let x_cross = (f64::from(prev_up[1]) - f64::from(curr_down[1]))
                    / (f64::from(curr_down[0]) - f64::from(prev_up[0]));
                let y_cross = (f64::from(curr_down[0]) * f64::from(prev_up[1])
                    - f64::from(prev_up[0]) * f64::from(curr_down[1]))
                    / (f64::from(curr_down[0]) - f64::from(prev_up[0]));
                if x_cross > x_max {
                    x_max = x_cross;
                };
                if y_cross > y_max {
                    y_max = y_cross;
                };
                prev_up = curr_up;
            };
            if next.is_none() {
                break;
            };
            curr_up = next.unwrap();
            curr_down = curr_up;
            next = iter.next();
        }

        if x_min == f64::MAX || y_min == f64::MAX || y_max == f64::MIN || x_max == f64::MIN {
            return 0f64;
        };
        (x_max - x_min) * (y_max - y_min)
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
