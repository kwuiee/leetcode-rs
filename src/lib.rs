#![allow(dead_code)]
use std::iter::Iterator;

pub struct Solution;

impl Solution {
    /// # Q1: dive board
    ///
    /// ## 题目
    /// 你正在使用一堆木板建造跳水板。有两种类型的木板，其中长度较短的木板长度为shorter，长度较长的木板长度为longer。
    /// 你必须正好使用k块木板。编写一个方法，生成跳水板所有可能的长度。
    ///
    /// ## 要求
    /// 返回的长度需要从小到大排列。
    ///
    /// ## 示例：
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
    ///
    /// ## Solution Test
    ///
    /// ```rust
    /// use leetcode_rs::Solution;
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

    /// # Q2: two sum
    ///
    /// ## 题目
    ///
    /// 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那两个
    /// 整数，并返回他们的数组下标。
    ///
    /// 你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
    ///
    /// ## 示例
    ///
    /// 给定 nums = [2, 7, 11, 15], target = 9
    ///
    /// 因为 nums[0] + nums[1] = 2 + 7 = 9
    /// 所以返回 [0, 1]
    ///
    /// ## Solution Test
    ///
    /// > time effecient
    ///
    /// ```rust
    /// use leetcode_rs::Solution;
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

    /// # Q2. two sum
    ///
    /// ## Solution Test
    ///
    /// > memory effecient
    ///
    /// ```rust
    /// use leetcode_rs::Solution;
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

    /// # Q3. find median sorted arrays
    ///
    /// ## 题目
    ///
    /// 给定两个大小为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。
    ///
    /// 请你找出这两个正序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
    ///
    /// 你可以假设 nums1 和 nums2 不会同时为空。
    ///
    /// ## 示例
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
    ///
    /// ## Solution Test
    ///
    /// ```rust
    /// use leetcode_rs::Solution;
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
    /// ## Leetcode Test
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
