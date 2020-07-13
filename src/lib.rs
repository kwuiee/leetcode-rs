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
        for i in 0..nums.len() {
            if let Some(v) = map.get(&(target - nums[i])) {
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
}
