---
type: 题库
status: 已发布
level: 进阶
topic:
  - 面试求职
  - 模型训练
---

# 算法题：传统算法与AI手撕分类版

> 共 503 道；传统算法沿用十二大类，AI/ML/训练推理/CUDA现场实现统一归入“AI手撕题”；已删除 45 条无具体题干的噪音记录。

## 数组与字符串

1. 【数组｜L1】怎么找某vector的倒数第二个元素？ 「百度」
2. 【[LC 14](https://leetcode.cn/problems/longest-common-prefix/)｜字典树、数组、字符串｜L1】手撕 最长公共前缀 「京东」
3. 【数组｜L1】为什么要加if(id<n)这种边界判断？ 「百度」
4. 【数组、矩阵｜L2】上三角矩阵，下三角矩阵 「百度」
5. 【前缀和、数组｜L2】前缀乘积。 「百度文心」
6. 【[LC 151](https://leetcode.cn/problems/reverse-words-in-a-string/)｜双指针、字符串｜L2】反转字符串的每个单词（后输出的单词先输出）。 「美团」
7. 【[LC 88](https://leetcode.cn/problems/merge-sorted-array/)｜数组、双指针、排序｜L2】合并两个有序数组（核心代码模式）。变体：合并后去重，如何保证时间复杂度？；另一场次补充：手撕代码：两个升序序列的合并，询问思路和复杂度。 「字节跳动、美团」
8. 【[LC 169](https://leetcode.cn/problems/majority-element/)｜数组、哈希表、分治、计数、排序｜L2】手撕：找出数组中出现次数大于数组长度一半的多数元素（LeetCode 169）。 「字节跳动、百度、美团」
9. 【字符串、设计｜L2】大量字符串拼接（如上万字符串）应该用什么方式？能否直接用“+”？用 string 以及并发的时间复杂度分别是多少？ 「百度、美团」
10. 【模拟、字符串｜L2】给一个字符串例如'abcde'，依次将第i个字符移到末尾，'abcde'->'bcdea'->'bdeac'->'bdace'->'bdaec'->'bdaec'。 「百度」
11. 【字符串、双指针｜L2】code:删除字符串中多余的空格，只保留一个。 「京东」
12. 【[LC 189](https://leetcode.cn/problems/rotate-array/)｜数组、数学、双指针｜L2】手撕代码：字符数组原地向右挪动K位（数组旋转）。 「拼多多」
13. 【数组｜L2】给定一个数组，找出数组中值最大和值第二大的两个数值，考虑时间复杂度尽可能低。 「百度」
14. 【[LC 228](https://leetcode.cn/problems/summary-ranges/)｜数组｜L2】实现汇总区间。 「字节跳动」
15. 【[LC 867](https://leetcode.cn/problems/transpose-matrix/)｜数组、矩阵、模拟｜L2】matrix transpose（矩阵转置） 「字节跳动」
16. 【数组｜L2】代码题：求数组连续出现相同数字的最大次数（一次遍历的easy题）。 「字节跳动」
17. 【字符串匹配、字符串｜L3】KMP算法是什么？解释其核心思想，包括next数组的构建和匹配过程。 「百度」
18. 【[LC 15](https://leetcode.cn/problems/3sum/)｜数组、双指针、排序｜L3】实现三数之和算法（LeetCode 15）及其变体：LeetCode 15 改成了3（具体变体未说明，保留原描述）。要求找出数组中所有和为0的三元组，不可重复。 「京东、百度、美团、腾讯、蚂蚁集团、阿里巴巴」
19. 【[LC 30](https://leetcode.cn/problems/substring-with-concatenation-of-all-words/)｜哈希表、字符串、滑动窗口｜L3】串联所有单词的子串。给定一个字符串s和一个单词列表words，找出s中所有可以由words中所有单词串联形成的子串的起始位置。 「字节跳动」
20. 【模拟、数组｜L3】给一个整数列表如[1,2,3,4]，依次加上符号变成1+2-3+4，然后如下放进列表:[1+2,2-3,3+4]即[3,-1,7]，一直到列表中只有最后一个数字，输出这个数字。 「百度」
21. 【双指针、字符串｜L3】实现删除字符串中的连续空格，要求O(n)时间复杂度、O(1)空间复杂度。 「阿里通义」
22. 【[LC 415](https://leetcode.cn/problems/add-strings/)｜数学、字符串、模拟｜L3】两数相加的字符串版本（LeetCode模式）。实现字符串相加。；另一场次补充：算法题：leetcode大数加法。 「字节跳动、美团、腾讯」
23. 【字符串、排序｜L3】实现字符串重排序之后的重合。 「腾讯」
24. 【[LC 498](https://leetcode.cn/problems/diagonal-traverse/)｜数组、矩阵、模拟｜L3】手撕对角线遍历矩阵。 「美团」
25. 【[LC 74](https://leetcode.cn/problems/search-a-2d-matrix/)｜数组、二分查找、矩阵｜L3】搜索二维矩阵：矩阵每行递增，且每行最后一个元素小于下一行第一个元素（类似力扣240），查找target是否存在。 「百度、美团」
26. 【数组、前缀和｜L3】特别数标识输出给一个数组，针对数组中每一个数字，如果它大于左侧所有数字且小于右侧所有数字，则对应位置标识为1，不满足任何一个条件则标识为0。 「百度」
27. 【[LC 5](https://leetcode.cn/problems/longest-palindromic-substring/)｜双指针、字符串、动态规划｜L3】手撕最长回文子串（LeetCode 5），要求输出子串，ACM格式，注意输入格式正确。 「京东、字节跳动、拼多多、百度、美团、阿里巴巴」
28. 【[LC 56](https://leetcode.cn/problems/merge-intervals/)/[LC 128](https://leetcode.cn/problems/longest-consecutive-sequence/)｜数组、排序、并查集、哈希表｜L3】code 原创题,并不难。像lc128最长连续序列和lc56合并区间的杂交版 「美团」
29. 【[LC 54](https://leetcode.cn/problems/spiral-matrix/)｜数组、矩阵、模拟｜L3】代码实现螺旋矩阵（顺时针遍历二维矩阵），即LeetCode 54题。 「字节跳动、美团、蚂蚁集团、阿里通义」
30. 【数组、动态规划｜L3】观光景点最高得分。 「百度」
31. 【数组、双指针｜L3】算法题：递增序列变体。 「美团」
32. 【递归、字符串｜L3】递归字符切分逻辑。 「美团」
33. 【[LC 209](https://leetcode.cn/problems/minimum-size-subarray-sum/)｜数组、二分查找、前缀和、滑动窗口｜L3】手撕：长度最小子数组问题（LeetCode 209/Hot100），返回最小长度。 「拼多多、百度」
34. 【[LC 238](https://leetcode.cn/problems/product-of-array-except-self/)｜数组、前缀和｜L3】代码：除自身外数组乘积（LeetCode 238/Hot100）。 「腾讯」
35. 【[LC 229](https://leetcode.cn/problems/majority-element-ii/)｜数组、哈希表、计数、排序｜L4】找出有序数组中所有出现次数严格大于n/3的数字，要求时间复杂度低于O(n)。 「字节跳动」
36. 【模拟、数组｜L4】有两个数组M和N，N代表一系列直径为M[i]的连续隧道，M代表一系列直径为N[i]的圆柱体，按顺序将一系列圆柱体往隧道里送，如果圆柱体直径大于某段隧道直径就会卡住，后续圆柱体也过不去，求最后圆柱体所在的最小位置，如果都通过了就返回-1。 「字节跳动」
37. 【数组、数学｜L4】给列表套列表套列表，问这个array的shape，再分别写一下对每个axis求和的结果是什么，对axis求和的数学依据是什么，为什么是这几个element求和，从数学角度怎么解释？ 「阿里通义」
38. 【[LC 391](https://leetcode.cn/problems/perfect-rectangle/)｜几何、数组、哈希表、数学、扫描线｜L4】力扣完美矩形变体（原题：判断多个矩形能否恰好覆盖一个大矩形，无重叠无空隙；变体可能增加条件或改变输出）。 「拼多多」
39. 【[LC 977](https://leetcode.cn/problems/squares-of-a-sorted-array/)｜数组、双指针、排序｜L4】给定数组 [-3, -2, -1, 0, 3, 6]，将其平方后排序。；另一场次补充：算法题：有序数组的平方（LeetCode 977）。 「腾讯」
40. 【数组、模拟｜L4】code:手写一下对给定axis求和（得分前面的维度和后面的维度，先flatten然后再求和的时候每次加stride）。 「阿里通义」
41. 【模拟、字符串｜L4】实现一段超长文本（可能几千万字）的切分，规则：1. 每段有最大长度K，段落以标点结束；2. 如果某段到了K还没遇到标点，就向后找最近的标点作为段尾（此时允许超过K）；3. 如果相邻几个短句加起来不超过K，要尽量合并成一段，让长度尽量接近K。；另一场次补充：手写简易文本Chunking分割算法，实现固定长度+语义约束的文本分块。 「阿里巴巴、阿里通义」
42. 【[LC 3](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)｜哈希表、字符串、滑动窗口｜L4】手撕无重复字符的最长子串（LeetCode 3，Hot100）。给定一个字符串s，找出其中不含有重复字符的最长子串的长度。变体：最长不同子数组的长度。；另一场次补充：给定一个字符串流，实现一个滑动窗口，返回当前窗口内的最长不重复子串长度。 「京东、字节Seed、字节跳动、美团、腾讯、腾讯混元、阿里巴巴」
43. 【[LC 76](https://leetcode.cn/problems/minimum-window-substring/)｜哈希表、字符串、滑动窗口｜L4】手撕最小覆盖子串：给定两个字符串s和t，请在s中找出包含t所有字符（出现次数也要满足）的最短子串。变体1：Code字符串中包含query字符的最短子串。变体2：给一个长度为L的数组，可能是m种颜色，找到最小的n，使得连续n个元素包含全部m种颜色。 「京东、美团、蚂蚁集团、阿里通义」
44. 【[LC 1769](https://leetcode.cn/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/)｜数组、字符串、前缀和｜L4】手撕leecode1769的变体，改成每移动一个站台需要一个能量 「小红书」
45. 【[LC 41](https://leetcode.cn/problems/first-missing-positive/)｜数组、哈希表｜L4】给定一个数组，找出缺失的最小正整数（LeetCode 41）。要求时间复杂度O(N)，空间复杂度O(1)。请讲下你的双指针slow/fast思路，并分析时间复杂度。 「字节Seed、字节跳动、百度」
46. 【[LC 340](https://leetcode.cn/problems/longest-substring-with-at-most-k-distinct-characters/)｜哈希表、字符串、滑动窗口｜L4】【LeetCode - 340】至多包含 K 个不同字符的最长子串。 「百度」

## 链表

1. 【[LC 21](https://leetcode.cn/problems/merge-two-sorted-lists/)｜递归、链表｜L2】Coding：归并排序两个有序链表 (LeetCode 21)。 「字节跳动、美团、腾讯混元」
2. 【[LC 141](https://leetcode.cn/problems/linked-list-cycle/)｜哈希表、链表、双指针｜L2】代码题/手撕：实现环形链表检测（LeetCode 141），判断链表是否有环。 「拼多多、百度、美团」
3. 【[LC 19](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)｜链表、双指针｜L3】删除链表的倒数第N个节点（LeetCode 19）：解释思路（如快慢指针或暴力法）。变体：返回倒数第K个节点，口述思路。 「百度、腾讯、腾讯混元」
4. 【[LC 237](https://leetcode.cn/problems/delete-node-in-a-linked-list/)｜链表｜L3】如何删除单链表的某个node（只给指向该node的指针而不是head）？ 「字节跳动」
5. 【[LC 82](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/)｜链表、双指针｜L3】删除链表中重复的元素II（LeetCode 82）：手撕代码。 「京东、字节跳动」
6. 【[LC 206](https://leetcode.cn/problems/reverse-linked-list/)/[LC 92](https://leetcode.cn/problems/reverse-linked-list-ii/)｜递归、链表｜L3】反转链表（LeetCode 206）：要求递归和迭代两种方式；包括反转链表II（部分反转，LeetCode 92）。 「字节跳动、百度、阿里巴巴、阿里通义」
7. 【[LC 142](https://leetcode.cn/problems/linked-list-cycle-ii/)｜哈希表、链表、双指针｜L3】长度为n的链表，判断是否回环，并返回回环的节点（LeetCode 142 环形链表II），如果没有回环，返回None。 「百度」
8. 【[LC 160](https://leetcode.cn/problems/intersection-of-two-linked-lists/)｜哈希表、链表、双指针｜L3】是否重合链表（LeetCode 160 相交链表）？ 「腾讯」
9. 【[LC 143](https://leetcode.cn/problems/reorder-list/)｜栈、递归、链表、双指针｜L3】重排链表（LeetCode 143）：手撕代码。 「百度、美团、阿里通义」
10. 【[LC 2](https://leetcode.cn/problems/add-two-numbers/)｜递归、链表、数学｜L3】链表两数相加（LeetCode 2）：需自己定义链表数据结构，使用双指针法，并分析时间复杂度和空间复杂度。 「字节跳动、百度、美团、腾讯、腾讯混元」
11. 【链表、设计｜L3】手撕链表相关的题，问时间复杂度，怎么优化。链表过长还有什么优化？ 「美团、腾讯」
12. 【链表、排序｜L3】按照node->val的绝对值排序链表。 「字节跳动」
13. 【[LC 25](https://leetcode.cn/problems/reverse-nodes-in-k-group/)｜递归、链表｜L4】K个一组反转链表（LeetCode 25）：先讲思路再写代码。变体：两两反转链表（即K=2）。变体：纸上写k组无序链表排序（即先对每组内排序再反转？需明确）。最后不足K个也要反转。要求手撕代码。；另一场次补充：K个一组翻转链表时，k=1或k大于链表长度应如何处理？ 「字节跳动、拼多多、百度、美团、腾讯、阿里巴巴」
14. 【[LC 138](https://leetcode.cn/problems/copy-list-with-random-pointer/)｜哈希表、链表｜L4】复制带随机指针的链表。 「蚂蚁集团」
15. 【[LC 148](https://leetcode.cn/problems/sort-list/)｜链表、双指针、分治、排序、归并排序｜L4】实现排序链表。手撕：排序链表。 「字节跳动」
16. 【链表｜L4】在实际开发中，链表的插入和删除操作虽然复杂度较低，但它是否有其他潜在的性能开销?比如在内存管理或其他方面，你怎么看? 「美团」
17. 【链表、双指针、哈希表｜L4】判断两个单链表是否相交（可能有环）；判断链表是否相交？ 「字节跳动、百度」

## 树与二叉树

1. 【[LC 144](https://leetcode.cn/problems/binary-tree-preorder-traversal/)｜栈、树、深度优先搜索、二叉树｜L1】二叉树先序遍历 「百度」
2. 【[LC 104](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)｜树、深度优先搜索、广度优先搜索、二叉树｜L1】二叉树的最大深度 「字节跳动」
3. 【二叉树、树｜L1】二叉树的遍历方式有哪些？并分别说明前序、中序、后序遍历是什么？中序遍历的结果有什么特点？ 「百度」
4. 【树、二叉树｜L1】搜索树的节点数量一般有多少？ 「美团」
5. 【树、二叉搜索树｜L2】为什么二叉搜索树会退化？最坏情况下会发生什么？如何避免二叉树退化？ 「百度」
6. 【[LC 102](https://leetcode.cn/problems/binary-tree-level-order-traversal/)｜树、广度优先搜索、二叉树｜L2】实现二叉树的层序遍历（LeetCode 102. 二叉树的层序遍历），手写代码，并分析时间复杂度和空间复杂度。 「字节Seed、字节跳动、美团、阿里巴巴」
7. 【树、深度优先搜索、广度优先搜索｜L2】多叉树查找值，时间复杂度要求最优，BFS/DFS都行。 「腾讯」
8. 【树、设计｜L3】范围查询（如between）在B+树中是如何执行的？ 「百度」
9. 【树、递归｜L3】扩展到 N 叉树怎么改。 「拼多多」
10. 【[LC 208](https://leetcode.cn/problems/implement-trie-prefix-tree/)｜设计、字典树、哈希表、字符串｜L3】实现Trie前缀树。支持插入、搜索和前缀搜索操作。 「阿里巴巴」
11. 【[LC 230](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)｜树、深度优先搜索、二叉搜索树、二叉树｜L3】搜索二叉树第k个大小的数。 「腾讯」
12. 【二叉搜索树、树｜L3】如果用普通二叉搜索树，范围查询怎么做? 「百度」
13. 【树、二叉树｜L3】二叉树的下一个节点(node有prev和next指针)。 「百度」
14. 【[LC 114](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)｜栈、树、深度优先搜索、链表、二叉树｜L3】手撕: 二叉树展开为链表（二叉树转为单链表）。变体：二分查找树转换为链表。 「百度、腾讯」
15. 【树、二叉树、二叉搜索树、递归｜L3】（追问）如何保证生成的树是平衡的？ 「快手」
16. 【[LC 1008](https://leetcode.cn/problems/construct-binary-search-tree-from-preorder-traversal/)｜栈、树、二叉搜索树、数组、二叉树、单调栈｜L3】手撕:先序遍历构建二叉搜索树(有复杂度要求)。 「美团」
17. 【[LC 226](https://leetcode.cn/problems/invert-binary-tree/)｜树、深度优先搜索、广度优先搜索、二叉树｜L3】反转二叉树（非递归反转二叉树）。 「腾讯、阿里通义」
18. 【树、二叉树、深度优先搜索｜L3】用你熟悉的语言（优先用C++）遍历一棵二叉树（不一定是满二叉），输出以每个结点为根、固定高度为K的子树。要求：每棵子树分别以深度优先顺序进行序列化。 「腾讯」
19. 【[LC 572](https://leetcode.cn/problems/subtree-of-another-tree/)｜树、深度优先搜索、二叉树、字符串匹配、哈希函数｜L3】Code:另一棵树的子树。判断一棵树是否是另一棵树的子树。 「阿里通义」
20. 【[LC 101](https://leetcode.cn/problems/symmetric-tree/)｜树、深度优先搜索、广度优先搜索、二叉树｜L3】算法题：对称二叉树。 「百度」
21. 【[LC 236](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)｜树、深度优先搜索、二叉树｜L3】二叉搜索树的最近公共祖先（LeetCode 236），并找到离给定两个节点最近的节点。 「京东、字节跳动、百度」
22. 【[LC 108](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)｜树、二叉搜索树、数组、分治、二叉树｜L3】手撕:将有序数组转换为二叉搜索树。 「快手」
23. 【树、哈希表｜L3】根据输入的数据(id，name，parentid)创建一棵树，越快越好，如果输入不对有报警机制。 「字节Seed」
24. 【广度优先搜索、树｜L3】算法题环节是一道树的层次遍历。 「美团」
25. 【[LC 129](https://leetcode.cn/problems/sum-root-to-leaf-numbers/)｜树、深度优先搜索、二叉树｜L3】LeetCode 129. 求根节点到叶节点数字之和。 「字节跳动、蚂蚁集团」
26. 【[LC 515](https://leetcode.cn/problems/find-largest-value-in-each-tree-row/)｜树、深度优先搜索、广度优先搜索、二叉树｜L3】二叉树每一层的最大值。 「百度」
27. 【树、二叉搜索树｜L3】讲一下红黑树，并说明红黑树和AVL树的时间复杂度是多少？ 「百度、腾讯混元」
28. 【[LC 112](https://leetcode.cn/problems/path-sum/)｜树、深度优先搜索、广度优先搜索、二叉树｜L3】给定一个二叉树和一个目标值，判断是否存在一条从根节点到叶子节点的路径，使得路径上所有节点值之和等于目标值。 「字节跳动、拼多多、百度」
29. 【树、二叉树、二叉搜索树｜L3】退化回链表的阈值是多少？ 「字节跳动」
30. 【[LC 105](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)｜树、数组、哈希表、分治、二叉树｜L3】根据前序和中序遍历重建二叉树，并说出后序遍历。 「百度、腾讯混元、阿里通义」
31. 【[LC 103](https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/)｜树、广度优先搜索、二叉树｜L3】LeetCode medium 103. 二叉树的锯齿形层序遍历。 「字节跳动」
32. 【[LC 199](https://leetcode.cn/problems/binary-tree-right-side-view/)｜树、深度优先搜索、广度优先搜索、二叉树｜L4】二叉树的左右视图，要求实现列表转二叉树（建树），并分别输出左视图和右视图，采用ACM模式（自己处理输入输出）。 「字节Seed、字节跳动」
33. 【其他算法｜L4】实现二叉树放水算法。 「MiniMax」
34. 【[LC 124](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)｜树、深度优先搜索、动态规划、二叉树｜L4】Code: 二叉树中的最大路径和（LeetCode 124，Hard）。手撕代码，找到二叉树中任意两个结点之间的最大路径和。 「字节跳动、拼多多、百度、美团、阿里巴巴、阿里通义」
35. 【[LC 440](https://leetcode.cn/problems/k-th-smallest-in-lexicographical-order/)｜字典树｜L4】字典序的第K小数字。给定整数n和k，找到1到n中字典序第k小的数字。 「字节跳动」
36. 【[LC 437](https://leetcode.cn/problems/path-sum-iii/)｜树、深度优先搜索、二叉树｜L4】手撕:路径总和III 「字节跳动」

## 图与搜索

1. 【深度优先搜索、广度优先搜索｜L1】coding 基础：DFS和BFS分别是什么？ 「百度」
2. 【深度优先搜索、树｜L2】找多级目录下的所有data.txt的路径 「百度」
3. 【图、最短路｜L3】A*算法相关：1. A*中的close和open指的是什么？2. A*算法的原理是什么？3. BFS，A*和Dijkstra的关系。4. 介绍A*和hyper A*的区别？5. 如何在自动驾驶中优化A*？6. 它和Dijkstra的区别是什么？7. 怎么从路由图上找最短路径？ 「百度、腾讯」
4. 【[LC 526](https://leetcode.cn/problems/beautiful-arrangement/)｜位运算、数组、动态规划、回溯、位掩码｜L3】写题: leetcode 526，优美的排列 「蚂蚁集团」
5. 【[LC 46](https://leetcode.cn/problems/permutations/)｜数组、回溯｜L3】实现不重复的全排列（LeetCode 46）。给定一个不含重复数字的数组，返回所有可能的全排列。；另一场次补充：给定正整数n，输出1-n的全排列 「MiniMax、字节Seed、字节跳动、百度、阿里巴巴」
6. 【[LC 47](https://leetcode.cn/problems/permutations-ii/)｜数组、回溯、排序｜L3】实现全排列II（LeetCode 47）。 「百度」
7. 【[LC 79](https://leetcode.cn/problems/word-search/)｜深度优先搜索、数组、字符串、回溯、矩阵｜L3】给定一个二维字符矩阵和一个字符串，判断该字符串是否在矩阵内，要求连续路径由上下左右相邻的单元格组成，且每个单元格只能使用一次。实现单词搜索函数。 「字节跳动、百度」
8. 【广度优先搜索、图｜L3】手撕图搜索传染题 「蚂蚁集团」
9. 【[LC 841](https://leetcode.cn/problems/keys-and-rooms/)｜深度优先搜索、广度优先搜索、图｜L3】LeetCode 841. 钥匙和房间。 「字节跳动、阿里巴巴」
10. 【[LC 93](https://leetcode.cn/problems/restore-ip-addresses/)｜字符串、回溯｜L3】手撕：给定一个只包含数字的字符串，求有多少种划分正确IP地址的方式。变体：给定一个字符串，如"2552325523"，进行正确的IP切分，返回["255.232.55.23", "255.23.255.23"]。 「美团、腾讯、阿里巴巴」
11. 【回溯、数组｜L3】手撕：给一个数组和一个目标值，返回所有和为目标值的子序列。 「京东」
12. 【[LC 200](https://leetcode.cn/problems/number-of-islands/)｜深度优先搜索、广度优先搜索、并查集、数组、矩阵｜L3】岛屿问题：1. 实现岛屿数量（LeetCode 200），要求分别用DFS和BFS实现，并说明并查集思路（代码可不写出）。2. 手撕岛屿最大面积（最大连通域的面积），要求用DFS或BFS遍历网格，求每片岛屿的面积。3. 找最大连通子图的大小。4. 手撕算法：子集型回溯问题。 「京东、字节Seed、字节跳动、拼多多、百度、美团、腾讯、阿里巴巴、阿里通义」
13. 【[LC 200](https://leetcode.cn/problems/number-of-islands/)/[LC 3](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)｜深度优先搜索、广度优先搜索、并查集、数组、矩阵、哈希表、字符串、滑动窗口｜L3】手撕代码：1. 岛屿数量（LeetCode 200）：给定一个由'1'（陆地）和'0'（水）组成的二维网格，计算岛屿的数量。2. 无重复字符的最长子串（LeetCode 3）：给定一个字符串，找出其中不含有重复字符的最长子串的长度。要求用滑动窗口或哈希表高效解决。 「美团、腾讯、腾讯混元、阿里巴巴」
14. 【[LC 207](https://leetcode.cn/problems/course-schedule/)｜深度优先搜索、广度优先搜索、图、拓扑排序｜L3】LeetCode 207. 课程表（中等）：给定一个项目列表和一个依赖关系列表（依赖关系是项目对的列表，其中第二个项目依赖于第一个项目），判断是否可能完成所有课程（即是否存在拓扑排序）。 「字节跳动、美团」
15. 【[LC 22](https://leetcode.cn/problems/generate-parentheses/)｜字符串、动态规划、回溯｜L3】手撕括号生成。括号生成（LeetCode 22）：生成n对有效括号的所有组合。；另一场次补充：代码：22.括号生成。 「字节跳动、腾讯、腾讯混元」
16. 【[LC 153](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)/[LC 200](https://leetcode.cn/problems/number-of-islands/)｜数组、二分查找、深度优先搜索、广度优先搜索、并查集、矩阵｜L3】手撕:旋转数组最小值(leetcode easy)，岛屿数量(leetcode middle)。 「腾讯」
17. 【最小生成树、图｜L3】coding 最小生成树算法 「蚂蚁集团」
18. 【[LC 77](https://leetcode.cn/problems/combinations/)｜回溯｜L3】组合(LeetCode 77)。 「阿里巴巴」
19. 【广度优先搜索、深度优先搜索、数组、矩阵｜L3】算法题：走迷宫问题(DFS/BFS) 给定一个二维网格，0表示可以通过，1表示障碍物，求从起点到终点的最短路径。 「字节跳动」
20. 【[LC 1197](https://leetcode.cn/problems/minimum-knight-moves/)｜广度优先搜索｜L3】手撕:马在m*n棋盘上从x0，y0走到x1，y1的最小步 「京东」
21. 【广度优先搜索、数组、矩阵｜L4】八数码问题：给定3*3数组，数字1-8和一个空位，每次交换空位与上下左右元素，判断能否转为目标状态。 「阿里通义」
22. 【深度优先搜索、图、广度优先搜索｜L4】coding是lc hot100图论题的变种（如岛屿数量、课程表等变种）以及对常用的深度学习库做了考察（如PyTorch、TensorFlow等）。 「美团」
23. 【[LC 694](https://leetcode.cn/problems/number-of-distinct-islands/)｜深度优先搜索、广度优先搜索、并查集、数组、哈希表、矩阵、排序、哈希函数｜L4】不同形状的岛屿数量。 「MiniMax」
24. 【[LC 329](https://leetcode.cn/problems/longest-increasing-path-in-a-matrix/)｜深度优先搜索、广度优先搜索、图、拓扑排序、记忆化、数组、动态规划、矩阵｜L4】二维矩阵中的最长递增路径，并输出该路径（序列）。给定一个整数矩阵，找出最长递增路径的长度，并输出路径。路径可以从任意单元格开始，每一步只能上下左右移动，且路径上的数字严格递增。 「京东、腾讯」
25. 【深度优先搜索、广度优先搜索、并查集、图｜L4】无向网格四连通判环。 「MiniMax」
26. 【[LC 621](https://leetcode.cn/problems/task-scheduler/)｜贪心、数组、哈希表、计数、排序、堆（优先队列）｜L5】拓扑排序实现任务调度器并用伪代码实现work-stealing的优化版本 「阿里巴巴」

## 动态规划

1. 【[LC 70](https://leetcode.cn/problems/climbing-stairs/)｜记忆化、数学、动态规划｜L1】实现爬楼梯（LeetCode 70），要求给出两种解法（如动态规划与递归/迭代），并写出代码。 「京东、百度、美团」
2. 【动态规划、数学｜L2】动态规划的原理和理解 (最优控制角度)。请从最优控制的角度解释动态规划的原理，包括最优子结构、重叠子问题、状态转移方程等核心概念。 「百度」
3. 【[LC 1143](https://leetcode.cn/problems/longest-common-subsequence/)｜字符串、动态规划｜L2】状态转移方程求max为什么不加一个dp[i-1][j-1]？请解释在动态规划中，例如求解最长公共子序列或编辑距离等问题时，状态转移方程中为什么有时不包含 dp[i-1][j-1] 项，而只考虑 dp[i-1][j] 和 dp[i][j-1] 的 max。 「腾讯」
4. 【[LC 509](https://leetcode.cn/problems/fibonacci-number/)｜递归、记忆化、数学、动态规划｜L2】手撕斐波那契数列。 「百度」
5. 【[LC 53](https://leetcode.cn/problems/maximum-subarray/)｜数组、分治、动态规划｜L2】实现最大子数组和（LeetCode 53），即求连续子数组的最大和。给定一个整数数组 nums，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。 「京东、字节跳动、百度、美团、蚂蚁集团」
6. 【[LC 62](https://leetcode.cn/problems/unique-paths/)｜数学、动态规划、组合数学｜L2】手撕代码矩阵走法：M行N列二维矩阵，左上角起点，右下角终点，每次只能向右或向下走一格，一共有多少种走法？ 「美团」
7. 【数组、动态规划｜L3】手撕：类似于子集划分，0-1背包问题。 「腾讯」
8. 【[LC 1567](https://leetcode.cn/problems/maximum-length-of-subarray-with-positive-product/)｜贪心、数组、动态规划｜L3】leetcode medium 1567. 乘积为正数的最长子数组长度。给定一个整数数组，找出乘积为正数的最长连续子数组的长度。 「字节跳动」
9. 【[LC 152](https://leetcode.cn/problems/maximum-product-subarray/)｜数组、动态规划｜L3】手撕:乘积最大子数组(LeetCode 152)。给定一个整数数组（包含正数和负数），找出乘积最大的连续子数组（至少包含一个数），返回最大乘积。要求使用动态规划实现。；另一场次补充：实现连续子数组的最大乘积和，并说明若改为求连续子数组的最大乘积，思路有何变化？；另一场次补充：code:最大乘积子数组？ 「京东、字节跳动、百度、腾讯、腾讯混元、蚂蚁集团、阿里巴巴、阿里通义」
10. 【[LC 416](https://leetcode.cn/problems/partition-equal-subset-sum/)｜数组、动态规划｜L3】代码：416.分割等和子集（0-1背包划分等和数组） 「字节跳动、腾讯」
11. 【[LC 139](https://leetcode.cn/problems/word-break/)｜字典树、记忆化、数组、哈希表、字符串、动态规划｜L3】单词拆分(LeetCode 139)。给定一个非空字符串s和一个包含非空单词列表的字典wordDict，判定s是否可以被空格拆分为一个或多个在字典中出现的单词。 「阿里巴巴」
12. 【动态规划、数学｜L3】圆环回原点问题。 「百度」
13. 【[LC 279](https://leetcode.cn/problems/perfect-squares/)｜广度优先搜索、数学、动态规划｜L3】给定一个正整数，求它最少能由几个完全平方数相加得到。例如，12 = 4 + 4 + 4，答案为3。手撕代码。 「百度、蚂蚁集团」
14. 【[LC 198](https://leetcode.cn/problems/house-robber/)｜数组、动态规划｜L3】手撕/实现打家劫舍算法。 「字节Seed、字节跳动、百度、阿里通义」
15. 【[LC 343](https://leetcode.cn/problems/integer-break/)｜数学、动态规划｜L3】整数拆分问题。变体：1. m个苹果放到n个盘子（可以为空），一共有多少种方法？2. 将n写成k个正整数之和的所有方案和方案数；3. 算法题：整数拆分。 「字节跳动、百度、腾讯混元」
16. 【[LC 1262](https://leetcode.cn/problems/greatest-sum-divisible-by-three/)｜贪心、数组、动态规划、排序｜L3】手撕:给一个正整数数组，找出里面能被3整除的和的最大值。 「腾讯」
17. 【[LC 221](https://leetcode.cn/problems/maximal-square/)｜数组、动态规划、矩阵｜L3】手撕 LeetCode 221：最大正方形。给定一个由 '0' 和 '1' 组成的二维矩阵，找出只包含 '1' 的最大正方形，并返回其面积。要求使用动态规划实现。 「字节Seed、百度、百度文心、美团」
18. 【[LC 64](https://leetcode.cn/problems/minimum-path-sum/)｜数组、动态规划、矩阵｜L3】二维dp：最短路径。给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和最小。每次只能向下或者向右移动一步。；另一场次补充：手撕最小路径和：在二维矩阵中，从右上角到左下角的最短加权路径和，中间还有-1的障碍物不能走。要求使用动态规划实现。 「京东、美团、腾讯」
19. 【字符串、动态规划｜L3】手撕：求最长公共子串。 「字节Seed、腾讯」
20. 【[LC 1143](https://leetcode.cn/problems/longest-common-subsequence/)｜字符串、动态规划｜L3】手撕：最长公共子序列（LCS），要求使用二维动态规划实现；最长公共子串。 「字节跳动、拼多多、百度、腾讯、腾讯混元、阿里通义」
21. 【数组、动态规划｜L3】手撕：最长公共子数组。 「美团」
22. 【[LC 516](https://leetcode.cn/problems/longest-palindromic-subsequence/)｜字符串、动态规划｜L3】最长回文子序列 「百度」
23. 【[LC 32](https://leetcode.cn/problems/longest-valid-parentheses/)｜栈、字符串、动态规划｜L3】实现最长有效括号（LeetCode 32）。 「字节跳动、百度」
24. 【[LC 213](https://leetcode.cn/problems/house-robber-ii/)｜数组、动态规划｜L3】Coding：从环中取不相邻的数字，数字和最大。 「字节Seed」
25. 【动态规划、数学、计数｜L3】手撕：一个环有10个点0-9，0开始出发，每步顺时针或逆时针走一个点，求经过n步回到0点有多少种不同走法。 「字节跳动」
26. 【[LC 91](https://leetcode.cn/problems/decode-ways/)｜字符串、动态规划｜L3】一条包含字母A-Z的消息通过'A' -> 1，'B' -> 2，...，'Z' -> 26方式进行了编码，给定一个只包含数字的非空字符串，求解码方法的总数。 「阿里巴巴」
27. 【[LC 322](https://leetcode.cn/problems/coin-change/)｜广度优先搜索、数组、动态规划｜L3】零钱兑换（LeetCode 322）：给定不同面额的硬币和一个总金额，计算凑成总金额所需的最少硬币个数。假设每种硬币的数量是无限的。变式：可能要求输出具体组合或处理其他约束。 「字节跳动、百度、腾讯、腾讯混元、蚂蚁集团、阿里巴巴」
28. 【[LC 518](https://leetcode.cn/problems/coin-change-ii/)｜数组、动态规划｜L3】代码：解决零钱兑换II（LeetCode 518），计算凑成目标金额的硬币组合数，要求区分顺序（组合问题）。 「腾讯、腾讯混元」
29. 【动态规划、数组｜L3】求数组的非连续最大和子序列，返回最大和以及子序列本身。 「阿里巴巴」
30. 【[LC 121](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)/[LC 122](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)/[LC 123](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/)/[LC 188](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/)｜数组、动态规划、贪心｜L4】手撕买卖股票全系列，包括经典股票四连问（LeetCode 121、122、123、188），要求实现代码并分析复杂度。变体：LeetCode 121（一次交易）、122（无限次交易）、123（两次交易）、188（k次交易）。另外包含买卖股票的最佳时机：LeetCode 121（一次交易）和122（多次交易）。 「字节跳动、百度、百度文心、美团、阿里巴巴」
31. 【[LC 337](https://leetcode.cn/problems/house-robber-iii/)｜树、深度优先搜索、动态规划、二叉树｜L4】leetcode medium 337. 打家劫舍III。 「字节跳动」
32. 【动态规划、数组｜L4】序列切分为前k个和n-k个，使得两部分方差和最大？给定一个序列，将其切分为前k个元素和后n-k个元素两部分，使得两部分方差之和最大。求最大方差和。 「字节跳动」
33. 【动态规划、二叉搜索树｜L4】场景题，动态规划二叉搜索。 「美团」
34. 【[LC 1458](https://leetcode.cn/problems/max-dot-product-of-two-subsequences/)｜数组、动态规划｜L4】两个数组的子序列求最大点积：给定两个数组，分别选取长度相同的子序列，计算对应位置乘积之和的最大值。 「字节跳动」
35. 【[LC 300](https://leetcode.cn/problems/longest-increasing-subsequence/)｜数组、二分查找、动态规划｜L4】手撕最长递增子序列（LeetCode 300）：找出数组的最长递增子序列。nums=[1,3,5,4,7]，返回[[1,3,4,7],[1,3,5,7]]。要求输出具体的一个最长递增子序列，并考虑优化（如O(n log n)解法）。；另一场次补充：最大递增子序列（动规，返回长度 -> 优化为返回整个序列 -> 优化相同长度的序列怎么选择更小的那一个）。给定一个整数数组，求最长严格递增子序列的长度，并输出该序列。若有多个相同长度的序列，选择字典序最小的一个。 「字节跳动、拼多多、百度、百度文心、腾讯、阿里巴巴、阿里通义」
36. 【[LC 718](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/)｜数组、二分查找、动态规划、滑动窗口、哈希函数、滚动哈希｜L4】Leetcode718最长重复子数组（要求小于O(n)）。 「字节Seed」
37. 【动态规划、组合数学｜L4】给一个n和k，代表从1-n选k个数字，计算得分，若选择i，而未选择i+1，则得一分。求最大得分。 「百度」
38. 【[LC 72](https://leetcode.cn/problems/edit-distance/)｜字符串、动态规划｜L4】实现编辑距离算法：给定两个字符串str1和str2，每次可选择插入、删除或替换一个字符，求最少操作次数使得两个字符串相同（LeetCode 72）。 「字节跳动、百度、美团、腾讯、蚂蚁集团、阿里巴巴」
39. 【[LC 44](https://leetcode.cn/problems/wildcard-matching/)｜贪心、递归、字符串、动态规划｜L4】手撕LeetCode Hard难度题目44通配符匹配（非Hot100原题）。实现通配符匹配，支持'?'匹配单个字符，'*'匹配任意字符序列（包括空序列）。 「百度」

## 排序与查找

1. 【数组、二分查找｜L1】有序数组插入的时间复杂度是多少？有序数组查找的时间复杂度是多少？ 「百度」
2. 【[LC 35](https://leetcode.cn/problems/search-insert-position/)｜数组、二分查找｜L2】算法题：搜索插入位置，二分查找。 「百度」
3. 【排序、数组、双指针｜L2】手撕代码环节是一道非常简单的去重归并排序。 「美团」
4. 【[LC 852](https://leetcode.cn/problems/peak-index-in-a-mountain-array/)｜数组、二分查找｜L2】山峰数组的索引（山脉数组的峰顶索引）。 「美团」
5. 【[LC 34](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)｜数组、二分查找｜L3】实现LC34：在排序数组中查找元素的第一个和最后一个位置（LeetCode 34）。给定一个按非递减顺序排列的整数数组和一个目标值，找出目标值在数组中的开始位置和结束位置。如果不存在，返回[-1, -1]。要求时间复杂度O(log n)。 「字节跳动、百度、阿里巴巴」
6. 【[LC 287](https://leetcode.cn/problems/find-the-duplicate-number/)｜位运算、数组、双指针、二分查找｜L3】给定长度为n+1的数组nums，其中元素取值范围为[1,n]，求唯一的重复数。 「字节Seed」
7. 【归并排序、数组、分治、树状数组｜L3】一个最小逆序对。给定一个数组，求逆序对的数量（即满足i<j且a[i]>a[j]的对数）。 「京东」
8. 【[LC 611](https://leetcode.cn/problems/valid-triangle-number/)｜贪心、数组、双指针、二分查找、排序｜L3】有效三角形的个数(LeetCode 611)。 「阿里巴巴」
9. 【堆（优先队列）、排序、数组｜L4】手撕：找出数组中最小的K个数，并按顺序返回。 变体1：使用最小堆实现。 变体2：给定候选商品列表，根据用户偏好做TopK筛选，要求时间复杂度低于O(N log N)。 「字节跳动、美团」
10. 【[LC 75](https://leetcode.cn/problems/sort-colors/)｜数组、双指针、排序｜L4】手撕三路排序。 「百度」
11. 【[LC 4](https://leetcode.cn/problems/median-of-two-sorted-arrays/)｜数组、二分查找、分治｜L4】给定两个大小分别为 m 和 n 的有序数组 nums1 和 nums2。请找出并返回这两个有序数组的中位数，要求时间复杂度为 O(log(m+n))。；另一场次补充：code: lc4 「字节跳动、阿里巴巴」
12. 【数组、分治、快速选择｜L4】手撕：无序数组中位数。 「腾讯」
13. 【排序、数组｜L4】手撕：给定一个随机数组，要求输出排序在中间的K个值。例如给定随机数组arr=[9,3,7,1,4]，当K=3时返回结果[3,7,4]或[3,4,7]；当K=1时返回结果[4]。给定的数组arr长度与K均为奇数。 「阿里巴巴」
14. 【[LC 704](https://leetcode.cn/problems/binary-search/)｜数组、二分查找｜L4】手撕二分查找代码，注意边界问题；二分查找变体：类似于插入位置问题，输入是json格式；二分查找的时间复杂度是多少？ 「字节跳动、百度、美团、腾讯」
15. 【[LC 56](https://leetcode.cn/problems/merge-intervals/)｜数组、排序｜L4】实现合并区间算法（LeetCode 56）。先讲思路，写代码，写输入输出并运行，分析时间复杂度。；另一场次补充：手撕代码：合并区间（LeetCode 56及其变种）。 「字节跳动、百度、腾讯、腾讯混元、阿里巴巴」
16. 【[LC 162](https://leetcode.cn/problems/find-peak-element/)｜数组、二分查找｜L4】手撕二分查找变种：给定一个数组，输出任意一个峰值，峰值定义为a[x] > a[x-1]且a[x] > a[x+1]，要求时间复杂度O(log n)。；另一场次补充：如何快速找出一个数组的一个峰值（峰值即比左右两个数都大）？ 「京东、字节跳动、阿里巴巴」
17. 【[LC 912](https://leetcode.cn/problems/sort-an-array/)｜数组、分治、桶排序、计数排序、基数排序、排序、堆（优先队列）、归并排序｜L4】手写归并排序代码。 「阿里巴巴」
18. 【排序、分治、递归｜L4】手写快速排序代码（原地排序），并分析时间复杂度。快速排序的原理是什么？如果不用递归的话怎么实现？手撕排序代码（快速排序）。 「字节Seed、字节跳动、百度、美团、腾讯、阿里巴巴」
19. 【排序、哈希表、数组｜L4】手撕代码：实现快速排序和两数之和。 「百度」
20. 【[LC 33](https://leetcode.cn/problems/search-in-rotated-sorted-array/)｜数组、二分查找｜L4】实现搜索旋转排序数组（LeetCode 33）：给定一个按升序排列的数组，在未知的某个点上进行了旋转，以及一个目标值，要求使用二分查找算法在数组中搜索目标值，返回其索引，若不存在则返回-1。；另一场次补充：实现力扣33题（搜索旋转排序数组）。 「字节跳动、百度、阿里巴巴、阿里通义」
21. 【[LC 81](https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/)｜数组、二分查找｜L4】手撕：LeetCode81 搜索旋转排序数组。假设数组原本按升序排列，但在某个未知点进行了旋转。给定一个目标值，如果目标值存在于数组中则返回true，否则返回false。数组可能包含重复元素。；另一场次补充：搜索旋转数组非常规。 「拼多多、百度」
22. 【[LC 378](https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/)｜二分查找、数组、矩阵、排序、堆（优先队列）｜L4】手撕：二维有序矩阵中找第k小的值（LeetCode 378）。要求时间复杂度O(n log n)（原记录未说明具体复杂度要求）。；另一场次补充：二维有序数组的第k个数 「字节跳动、百度、腾讯、阿里巴巴」
23. 【[LC 215](https://leetcode.cn/problems/kth-largest-element-in-an-array/)｜数组、分治、快速选择、排序、堆（优先队列）｜L4】手撕代码：找出无序数组中第K大的元素（LeetCode 215）。要求实现快速选择（QuickSelect）算法，并分析时间复杂度（平均O(n)，最坏O(n^2)）。可选的优化方法包括使用堆排序（堆排）或局部快排。另外，实现堆排序算法；求数组中的第K个最大元素，要求比快排更快，使用堆排序或优先队列。 「京东、字节跳动、百度、百度文心、腾讯、阿里巴巴、阿里通义」

## 贪心与双指针

1. 【[LC 1221](https://leetcode.cn/problems/split-a-string-in-balanced-strings/)｜贪心、字符串、计数｜L2】分割平衡字符串：在一个平衡字符串中，'L'和'R'字符数量相同。给定一个平衡字符串，将其分割成尽可能多的平衡子串，返回最大分割数量。 「字节跳动」
2. 【[LC 670](https://leetcode.cn/problems/maximum-swap/)｜贪心、数学｜L2】交换正整数的两位数字，使其尽可能大。给定一个正整数，最多可以交换一次任意两位数字，返回能得到的最大整数。 「字节跳动」
3. 【[LC 31](https://leetcode.cn/problems/next-permutation/)｜数组、双指针｜L3】实现LeetCode 31题：下一个排列。给定一个整数数组（表示一个数字），找到该数字的下一个字典序更大的排列。如果不存在下一个更大的排列，则将数组重新排列为最小的升序排列。要求原地修改，仅使用常数额外空间。 「字节Seed、百度」
4. 【[LC 167](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/)｜数组、双指针、二分查找｜L3】实现排序数组中和为给定值的数组对数，要求时间复杂度O(n)，空间复杂度O(1)。；另一场次补充：实现两数之和2（LeetCode 167）。 「百度、阿里通义」
5. 【[LC 621](https://leetcode.cn/problems/task-scheduler/)｜贪心、数组、哈希表、计数、排序、堆（优先队列）｜L3】手撕：任务调度器（LeetCode 621）：给定一个用字符数组表示的CPU需要执行的任务列表，以及一个冷却时间n，每个单位时间可以完成一个任务，相同任务之间必须间隔至少n个单位时间，求完成所有任务所需的最短时间。 「阿里巴巴、阿里通义」
6. 【[LC 134](https://leetcode.cn/problems/gas-station/)｜贪心、数组｜L3】加油站算法：在一条环路上有N个加油站，每个加油站有汽油gas[i]，从第i个加油站到第i+1个需要消耗cost[i]，求从哪个加油站出发可以走完一圈，否则返回-1。 「百度」
7. 【[LC 15](https://leetcode.cn/problems/3sum/)/[LC 27](https://leetcode.cn/problems/remove-element/)/[LC 167](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/)｜数组、双指针、排序、二分查找｜L3】实现双指针算法：给定一个有序数组，使用双指针法解决两数之和、三数之和或移除元素等问题。具体题目未指定，需现场实现。 「美团」
8. 【滑动窗口、数组｜L3】一个数组，找出一个最大连续子串，要求子串中每个元素都小于T。 「百度」
9. 【[LC 3](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)｜哈希表、字符串、滑动窗口｜L3】手撕代码：最长无重复子序列。 「字节跳动」
10. 【[LC 424](https://leetcode.cn/problems/longest-repeating-character-replacement/)/[LC 1004](https://leetcode.cn/problems/max-consecutive-ones-iii/)｜哈希表、字符串、滑动窗口、数组、二分查找、前缀和｜L3】一个数组只有a和b，有m次机会能把a换成b或b换成a，变化之后的最长连续相同字符的长度是多少？ 「腾讯」
11. 【[LC 165](https://leetcode.cn/problems/compare-version-numbers/)｜双指针、字符串｜L3】实现比较版本号。 「字节跳动」
12. 【[LC 2554](https://leetcode.cn/problems/maximum-number-of-integers-to-choose-from-a-range-i/)｜贪心、数组、哈希表、二分查找、排序｜L3】手撕：Maximum Number of Integers to Choose From a Range I 「百度」
13. 【[LC 55](https://leetcode.cn/problems/jump-game/)｜贪心、数组、动态规划｜L3】跳跃游戏：给定一个非负整数数组，初始位置为第一个索引，每个元素代表在该位置可以跳跃的最大长度，判断是否能到达最后一个索引。变体：手撕一个贪心策略的问题（跳跃游戏）。；另一场次补充：手撕算法:跳跃数组。 「字节跳动、百度、蚂蚁集团、阿里通义」
14. 【双指针、数组｜L4】手撕：一个数组先递增后递减，找出不重复的元素个数，要求常数空间复杂度，线性时间复杂度。 「百度」
15. 【[LC 767](https://leetcode.cn/problems/reorganize-string/)｜贪心、哈希表、字符串、计数、排序、堆（优先队列）｜L4】输入一个序列，输出调整后的序列，目标是将序列中相同的连续的元素打散隔开，尽量保证原有的序列变化不大。示例: 1322254 -> 1325242，132225 -> 132522。 「阿里巴巴」

## 栈队列与堆

1. 【[LC 94](https://leetcode.cn/problems/binary-tree-inorder-traversal/)｜栈、树、深度优先搜索、二叉树｜L3】实现二叉树的中序遍历非递归。 「字节跳动、百度、阿里巴巴」
2. 【[LC 347](https://leetcode.cn/problems/top-k-frequent-elements/)｜数组、哈希表、分治、桶排序、计数、快速选择、排序、堆（优先队列）｜L3】手撕前K个高频元素。 「字节跳动」
3. 【[LC 227](https://leetcode.cn/problems/basic-calculator-ii/)｜栈、数学、字符串｜L3】code: 字符串四则运算（LeetCode 227）。实现一个基本计算器来计算一个简单的字符串表达式的值，表达式包含非负整数和加减乘除运算符，不含括号。 「阿里巴巴」
4. 【[LC 20](https://leetcode.cn/problems/valid-parentheses/)｜栈、字符串｜L3】实现有效的括号（LeetCode 20）。 「字节跳动、百度、美团、阿里巴巴」
5. 【[LC 84](https://leetcode.cn/problems/largest-rectangle-in-histogram/)｜栈、数组、单调栈｜L3】柱状图中最大的矩形（LeetCode 84）。 「百度」
6. 【堆（优先队列）｜L3】PriorityQueue了解吗，有没有用过？ 「美团」
7. 【设计、链表、数组｜L3】假设我们在一个高频插入和删除操作的场景中，比如实现一个队列或栈，你会选择数组还是链表？ 「美团」
8. 【单调栈、栈｜L3】实现典型单调栈。 「拼多多」
9. 【[LC 394](https://leetcode.cn/problems/decode-string/)｜栈、递归、字符串｜L3】实现字符串解码（LeetCode 394），原记录未说明输入格式和输出要求。 「百度、阿里巴巴、阿里通义」
10. 【[LC 232](https://leetcode.cn/problems/implement-queue-using-stacks/)｜栈、设计、队列｜L3】实现队列：1. 用栈实现队列（LeetCode 232）。2. 手撕实现queue（不限方式）。 「字节跳动」
11. 【[LC 32](https://leetcode.cn/problems/longest-valid-parentheses/)｜栈、字符串、动态规划｜L3】手撕两道：最长有效括号和二叉树层序遍历。 「美团」
12. 【堆（优先队列）｜L3】最小堆底层用什么数据结构实现？ 「美团」
13. 【[LC 239](https://leetcode.cn/problems/sliding-window-maximum/)｜队列、数组、滑动窗口、单调队列、堆（优先队列）｜L3】滑动窗口最大值（一维），要求O(n)时间复杂度；部分场次要求取第k大的元素；ACM模式，从头写C++。设计如何利用堆最快的实现。实现队列的最大值。；另一场次补充：一个裁判打分，用滑动窗口记录最大值，最小值。给定一个数组和一个窗口大小k，计算每个滑动窗口内的最大值和最小值。 「京东、字节跳动、百度、腾讯、阿里巴巴」
14. 【[LC 150](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)｜栈、数组、数学｜L3】逆波兰表达式求值。 「百度」
15. 【[LC 739](https://leetcode.cn/problems/daily-temperatures/)｜栈、数组、单调栈｜L3】实现每日温度算法。 「字节跳动」
16. 【[LC 23](https://leetcode.cn/problems/merge-k-sorted-lists/)｜链表、分治、堆（优先队列）、归并排序｜L4】合并K个升序链表（LeetCode 23. Merge k Sorted Lists）。要求使用堆（heapq）的方法，并给出完整实现。多个链表合并。 「字节跳动、百度、腾讯、阿里巴巴」
17. 【[LC 862](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/)｜队列、数组、二分查找、前缀和、滑动窗口、单调队列、堆（优先队列）｜L4】和至少为K的最短子数组长度。 「字节跳动」
18. 【堆（优先队列）、数组、排序｜L4】有n个长度为m的升序数组，找出所有数中的前K大（TopK）。 「字节跳动」
19. 【[LC 42](https://leetcode.cn/problems/trapping-rain-water/)｜栈、数组、双指针、动态规划、单调栈｜L4】实现接雨水算法（LeetCode 42）。 「百度、美团、阿里巴巴」
20. 【[LC 853](https://leetcode.cn/problems/car-fleet/)｜栈、数组、排序、单调栈｜L4】LeetCode 853. 车队（中等）。 「字节跳动」
21. 【[LC 331](https://leetcode.cn/problems/verify-preorder-serialization-of-a-binary-tree/)｜栈、树、字符串、二叉树｜L4】手撕:lc331 验证二叉树的前序序列化 问复杂度，优化方法 「字节Seed」
22. 【[LC 1776](https://leetcode.cn/problems/car-fleet-ii/)｜栈、数组、数学、单调栈、堆（优先队列）｜L5】LeetCode Hard 1776. 车队II。 「字节跳动」

## 哈希与集合

1. 【设计、哈希表｜L1】1. HashMap与Map的区别？2. Map和Set的区别？ 「百度」
2. 【设计、哈希表｜L2】map 的 key 和 value 应该如何设计? 「百度」
3. 【哈希表、树、二叉搜索树｜L2】map和multimap; set和multiset 「百度」
4. 【[LC 1](https://leetcode.cn/problems/two-sum/)｜数组、哈希表｜L2】给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出和为目标值的那两个整数，并返回它们的数组下标。；另一场次补充：手撕很简单，leetcode 001 two sum 「京东、美团、腾讯、腾讯混元、阿里巴巴」
5. 【[LC 1](https://leetcode.cn/problems/two-sum/)｜数组、哈希表｜L2】代码题：一组正整数数组中有多少两数和为K的组合 「字节跳动」
6. 【哈希表、字符串｜L2】算法题：删除重复字符串。 「京东」
7. 【设计、哈希表、树、链表｜L2】HashMap底层数据结构是什么？具体包括Java HashMap和STL中Map、HashTable的底层实现数据结构。HashMap什么时候从链表转红黑树？为什么要满足数组长度 > 64？ 「字节跳动、美团、阿里巴巴」
8. 【[LC 349](https://leetcode.cn/problems/intersection-of-two-arrays/)/[LC 350](https://leetcode.cn/problems/intersection-of-two-arrays-ii/)｜数组、哈希表、双指针、二分查找、排序｜L2】两个数组的公共部分（数组交集）。 「腾讯」
9. 【哈希表、字符串｜L2】代码：找出重复最多的字符串，并返回字符串和重复次数。 「百度」
10. 【哈希表、设计｜L2】如果要存储一批商品并判断是否存在，应该用什么数据结构? 「百度」
11. 【[LC 560](https://leetcode.cn/problems/subarray-sum-equals-k/)｜数组、哈希表、前缀和｜L3】LeetCode 560. 和为K的子数组。给定一个整数数组和一个整数K，统计并返回该数组中和为K的连续子数组的个数。K可取0。变体：找到所有满足条件的子数组并返回起始与终止位置。示例数组[2,1,4,-1,5,0]，K值未注明。 「字节跳动、百度、蚂蚁集团、阿里巴巴」
12. 【[LC 974](https://leetcode.cn/problems/subarray-sums-divisible-by-k/)｜数组、哈希表、前缀和｜L3】LeetCode 974. 和可被K整除的子数组。给定一个整数数组和一个整数K，返回和可被K整除的连续子数组的个数。 「腾讯」
13. 【设计、哈希表｜L3】如何实现一个Map？ 「腾讯」
14. 【[LC 128](https://leetcode.cn/problems/longest-consecutive-sequence/)｜并查集、数组、哈希表｜L3】实现最长连续数字串（LeetCode 128 最长连续序列）。 「百度、腾讯」
15. 【[LC 299](https://leetcode.cn/problems/bulls-and-cows/)｜哈希表、字符串、计数｜L3】算法题：猜数字。 「百度」
16. 【哈希表、数组｜L3】如何快速对一个list去重，如果list的长度非常长(且不改变数据格式)？ 「百度」
17. 【哈希表、字符串、计数｜L4】大数据场景下，统计字符串表中出现的单词个数（裂开）；在大规模（几百万）数据中做高效检索。 「拼多多、阿里巴巴」

## 数学与位运算

1. 【数学｜L1】完美数是什么？ 「阿里通义」
2. 【[LC 7](https://leetcode.cn/problems/reverse-integer/)｜数学｜L1】输入123输出321（整数反转）。 「字节跳动」
3. 【递归、数学｜L1】用递归法求1到n的和。 「京东」
4. 【数学、位运算｜L2】手撕：写个函数实现乘法运算。 「京东」
5. 【[LC 191](https://leetcode.cn/problems/number-of-1-bits/)｜位运算、分治｜L2】给出数字返回内存为1的个数。 「腾讯」
6. 【[LC 461](https://leetcode.cn/problems/hamming-distance/)｜位运算｜L2】LeetCode: 计算两个整型数的Hamming距离，原型: int hammingDistance(int x, int y); 「腾讯」
7. 【[LC 384](https://leetcode.cn/problems/shuffle-an-array/)｜设计、数组、数学、随机化｜L2】手撕题1:扑克牌打乱（洗牌算法） 「字节跳动」
8. 【数学、数论｜L2】手撕:求k的分解质因数。 「百度」
9. 【数学、几何｜L2】手写欧氏距离、曼哈顿距离，对比三种向量距离算法的适用场景。 「阿里通义」
10. 【数学、数论｜L2】1000!有多少个0？ 「美团」
11. 【随机化、数学｜L2】随机数的生成。 「百度」
12. 【数学、概率与统计｜L2】用随机数函数randf()表示概率，如从一个数组中取出某个数的概率为0.5。 「字节跳动」
13. 【随机化、概率与统计｜L2】现场看了python random.choices和random.sample的文档，问有什么区别，从概率角度上来说表达式有什么不同？ 「百度」
14. 【递归、数学、字符串｜L3】NOIP1998复赛普及组第一题：2的幂次方表示。 「字节跳动」
15. 【位运算｜L3】对于一个很长的数，其中只有个别位是1，如何优化查找这些1的位置？问题转化为如何找到最低位的1？ 「腾讯」
16. 【几何、数学｜L3】每个点到稠密折线段的距离 「百度」
17. 【[LC 593](https://leetcode.cn/problems/valid-square/)｜几何、数学｜L3】手撕一道简单的哈希问题，给出点序列，求解总共能匹配到多少个正方形。 「拼多多」
18. 【[LC 69](https://leetcode.cn/problems/sqrtx/)｜数学、二分查找｜L3】实现平方根计算。变体：1. 手写IoU和开根号小数点后三位（不调包）；2. 实现sqrt(x)，向下取整；3. 非负整数求根号(二分法)；4. 多态的方式实现开根号；5. 实现开平方根；6. 求浮点数平方根（牛顿法/二分法），手写牛顿迭代法，收敛阈值怎么设置，该方法优势？7. 计算一个数的算术平方根，精度0.00001。 「字节跳动、百度、腾讯、阿里巴巴」
19. 【[LC 593](https://leetcode.cn/problems/valid-square/)｜几何、数学｜L3】代码：有效的正方形。 「百度」
20. 【数学、设计｜L3】手写有理数类，实现加减法和约分->找最大公约数，定义分子和分母、函数书写的位置。 「美团」
21. 【概率、序列｜L3】手撕: M个行为N个奖励，都是概率。求一个指定奖励序列的概率。 「字节跳动」
22. 【[LC 50](https://leetcode.cn/problems/powx-n/)｜递归、数学｜L3】浮点数的n次方。 「拼多多」
23. 【几何、数学｜L3】判断一个点是否在凸多边形内部和凹多边形内部。 「百度」
24. 【数学、计数、数组｜L3】给质数数组和一个k，从数组中选择两个数组成真分数，有多少组合方式大于k？ 「字节跳动」
25. 【数学、数组、矩阵｜L3】手撕代码：计算m×d矩阵两两之间的欧式距离（复杂度不超过m^2）。 「腾讯」
26. 【二分查找、数学｜L3】写一个开三次根号函数，对一个数求立方根，误差在1e-5。 「字节跳动、拼多多」
27. 【[LC 1823](https://leetcode.cn/problems/find-the-winner-of-the-circular-game/)｜递归、队列、数组、数学、模拟｜L3】破冰游戏（约瑟夫环）。 「字节跳动」
28. 【数学、模拟｜L3】如何利用计算机求π，不限时间复杂度，可以循环一亿次？ 「阿里巴巴」
29. 【数学、模拟｜L3】求a/b的商和余数，不能用除法和取余，要求使用两种方法，其中一种方法用加减实现整除。 「阿里巴巴」
30. 【[LC 470](https://leetcode.cn/problems/implement-rand10-using-rand7/)｜数学、拒绝采样、概率与统计、随机化｜L3】使用 Rand7() 实现 Rand10()（LeetCode 470）。 「字节跳动、百度、阿里巴巴、阿里通义」
31. 【几何、排序、数学｜L4】三维点凸包计算及凸包算法（Graham扫描法）。 「腾讯」
32. 【数学、贪心｜L4】小于n的最大值（leetcode无原题）。 「字节跳动」
33. 【[LC 528](https://leetcode.cn/problems/random-pick-with-weight/)｜数组、数学、二分查找、前缀和、随机化｜L4】给定一个长度为n的数组weights，其中weights[i]是元素i的非负权重。现在需要进行大量、近似无限次的有放回采样，如何让单次采样的时间复杂度尽可能低？实现带权随机采样算法。；另一场次补充：按概率采样。变体：1. 实现一个函数，给定一个nums数组和一个概率数组，根据概率返回nums中的数；2. 曝光概率区间均匀采样。 「字节跳动、百度、腾讯」
34. 【数学、动态规划、计数｜L4】有一个长度为n的序列a1, a2, ..., an，我们希望选择一个最大公约数不为1的子序列，求可以选择的最长子序列长度，以及这种最长的子序列总共有多少种。如果两个子序列所包含的元素值的多重集合相同，则认为它们是同一种方案，即不区分下标位置，仅按所含数字及出现次数判断是否相同。 「阿里巴巴」
35. 【[LC 1755](https://leetcode.cn/problems/closest-subsequence-sum/)｜位运算、数组、双指针、动态规划、位掩码、排序｜L4】最接近目标值的子序列和：给定一个整数数组和一个目标值，找出子序列的和，使其最接近目标值。返回该和。 「阿里通义」
36. 【[LC 398](https://leetcode.cn/problems/random-pick-index/)｜水塘抽样、哈希表、数学、随机化｜L4】最长流式随机采样。实现一个流式随机采样算法，从数据流中均匀随机选取k个样本（蓄水池采样）。 「百度」
37. 【[LC 553](https://leetcode.cn/problems/optimal-division/)｜数组、数学、动态规划｜L4】给一个整数数组a，对a中的相邻整数进行浮点除法，如a=[2,3,4]，即为2/3/4。在任意位置添加任意数目的括号，来改变算数的优先级，使得值最大，输出表达式。 「百度」

## 算法复杂度

1. 【二叉搜索树、树｜L1】二叉搜索树查找的时间复杂度是多少？ 「百度」
2. 【排序｜L1】排序有哪些算法，哪些是稳定的呢？ 「百度」
3. 【数组、链表、设计｜L1】请解释数组和链表的主要区别，以及它们各自适合的应用场景。 「美团」
4. 【设计｜L2】vector与list的区别与应用？ 「百度」
5. 【设计｜L2】STL中vector扩容为什么要以1.5倍或者2倍扩容？ 「百度」
6. 【其他算法｜L2】分析给定代码的时间复杂度和空间复杂度，并描述解题思路。具体追问：到底是哪一步产生了 O(n^2)？这个复杂度是怎么算出来的？为什么是这个复杂度？ 「京东、字节跳动、快手、百度、美团、阿里巴巴、阿里通义」
7. 【堆（优先队列）、排序｜L4】怎么样才能让堆排序稳定呢？ 「百度」
8. 【其他算法｜L4】如何考虑降低代码的复杂度？如果数据规模非常大，有没有优化思路？ 「百度、阿里巴巴」
9. 【排序、堆（优先队列）｜L4】堆排序是稳定的吗？ 「百度」
10. 【其他算法｜L4】如果数组本身有序，是否可以优化？ 「美团」

## 综合编程题

1. 【其他算法｜L1】if __name__ == '__main__'的作用是什么？ 「阿里巴巴」
2. 【其他算法｜L1】i++和++i的区别？ 「百度」
3. 【综合编程｜L1】Python Collection了解吗？ 「美团」
4. 【其他算法｜L1】Python中help以及dir的用法。 「美团」
5. 【其他算法｜L1】列表、元组是否可变？ 「阿里巴巴」
6. 【其他算法｜L1】了解友元概念吗？ 「腾讯」
7. 【综合编程｜L1】你熟练掌握哪些深度学习框架(PyTorch/TensorFlow)? 「京东」
8. 【其他算法｜L1】哪种成员变量在类的对象中实现共享？ 「百度」
9. 【其他算法｜L2】四种强制类型转换是哪些？ 「百度」
10. 【设计｜L2】类的成员函数可以当模板吗？ 「百度」
11. 【综合编程｜L2】DataLoader、Dataset、Sample的区别是什么？ 「百度」
12. 【综合编程｜L2】defer能不能修改函数返回值？ 「百度」
13. 【综合编程｜L2】delete和delete[]的区别 「百度」
14. 【设计、模拟｜L2】make_unique的实现原理 「百度」
15. 【综合编程｜L2】numpy和数组的区别、numpy的好处和原因，以及numpy对比python原生的列表或接口的优势 「美团」
16. 【综合编程｜L2】pandas筛选数据的题目。 「百度」
17. 【其他算法｜L2】Python的**是干什么的，**kwargs是干嘛的，@字符是干嘛的？ 「字节Seed」
18. 【多线程、设计、哈希表｜L2】Python有没有真正的多线程？python编程：多线程、多进程的库；list怎么快速去重？八股python怎么实现多线程。 「字节跳动、百度、美团」
19. 【设计｜L2】知道Python的装饰器么？装饰器的原理是什么？ 「字节Seed、阿里巴巴」
20. 【综合编程｜L2】介绍python解释器；循环和递归的区别 「字节Seed」
21. 【综合编程｜L2】八股python如何读取大文件 「美团」
22. 【其他算法｜L2】python里面的闭包是什么，with操作是什么，有什么好处？ 「腾讯」
23. 【其他算法｜L2】为什么用view？ 「字节跳动」
24. 【设计、模拟｜L2】说出pytorch中维度变换的函数 「京东」
25. 【设计、其他算法｜L2】可以用static变量来实现吗？ 「百度」
26. 【其他算法｜L2】如果输入列表中可能包含重复元素，如何避免重复解？ 「美团」
27. 【其他算法｜L2】右值引用有什么作用? 「腾讯」
28. 【设计｜L2】回调函数的作用 「百度」
29. 【设计｜L2】基础问答：Coding(复杂任务如何指挥 cc)、Python(Python list 和 tuple 区别？ 「百度文心」
30. 【其他算法｜L2】如果中间步骤报错了怎么做异常处理? 「蚂蚁集团」
31. 【并发、锁｜L2】悲观锁代码怎么写？ 「字节跳动」
32. 【设计、数组、链表、栈、队列｜L2】数据结构。要会:数组/链表/栈/队列/树/图/堆/哈希表;可选:线段树/树状数组/并查集/字典树等 「阿里巴巴」
33. 【设计｜L2】在什么场景下选择数组，什么场景下选择树? 「百度」
34. 【其他算法｜L2】如果一个方法抽取是只在一个场景中用的，现在以及未来都不会用，那需要抽取这个方法吗? 「美团」
35. 【并发、死锁｜L2】多线程写一个死锁。 「字节跳动」
36. 【设计｜L2】深拷贝与浅拷贝的区别？ 「百度、阿里巴巴」
37. 【其他算法｜L2】根据相似度召回top1的文本（给好了内置函数，只是要把代码根据逻辑写好）。 「腾讯」
38. 【其他算法｜L2】完整类型和非完整类型 「百度」
39. 【并发、线程通信｜L2】手撕代码：两个线程交替打印0-100 「拼多多」
40. 【并发、设计模式｜L2】实现一个线程安全的单例模式(构造方法需要被声明为private类型) 「阿里通义」
41. 【综合编程｜L2】你熟练的编程语言(如Python、C++)在计算机视觉中有什么具体应用? 「京东」
42. 【其他算法｜L2】构造函数，析构函数，静态成员函数，友元函数，哪些可以设置为virtual？ 「百度」
43. 【其他算法｜L2】野指针问题？如何避免？如何检测？ 「百度」
44. 【多目标跟踪、数据关联、代码解析｜L3】deepsort代码解析。 「腾讯」
45. 【推荐系统、MIND｜L3】讲一下MIND的具体实现 「阿里巴巴」
46. 【[LC 177](https://leetcode.cn/problems/nth-highest-salary/)｜数据库｜L3】编写一个SQL查询，获取Employee表中第N高的薪水。 「百度」
47. 【数据库｜L3】手撕SQL题。 「阿里通义」
48. 【设计｜L3】实现std::move，使用模版函数，namespace。 「百度」
49. 【设计、数组｜L3】STL vector的实现，删除其中的元素，迭代器如何变？空间都是连续的情况下，就可以实现内存空间复用化？ 「百度」
50. 【设计、数组｜L3】C++ vector相关问题：1. vector容器的用法和扩容机制；2. 为什么每次扩容是两倍增长，而不是3倍或其他数值？3. emplace_back和push_back的区别。 「字节跳动、百度」
51. 【系统设计、代码审查｜L3】做一个代码审查系统。 「蚂蚁集团」
52. 【设计、模拟｜L3】会议室预定问题：判断某个时刻点是否可以预定，查询最近可以预定的时间。 「腾讯」
53. 【设计、多线程、设计模式、并发｜L3】手写单例模式：饿汉式、懒汉式；他们有什么区别？；另一场次补充：手撕:单例模式 (双重校验锁和静态内部类)。 「京东、字节跳动、百度」
54. 【多目标跟踪、数据关联、代码实践｜L3】多目标跟踪中的数据关联代码实践。 「腾讯」
55. 【其他算法｜L3】完美转发std::forward。 「百度」
56. 【其他算法｜L3】如果有个常量被很多地方共用，你怎么判断能不能融？ 「阿里巴巴」
57. 【其他算法｜L3】析构函数抛出异常会导致什么情况？ 「百度」
58. 【设计、树、有序集合｜L3】如何设计一种数据结构，既支持快速插入又支持范围查询？请用Java代码实现，并说明如何设计座位和区间。 「百度、蚂蚁集团」
59. 【设计｜L3】灵活的数据集抽象? 「阿里巴巴」
60. 【设计、哈希表、树｜L3】请介绍 List、Set、Sorted Set 及其典型应用场景。 「美团」
61. 【其他算法｜L3】为什么泛型函数也需要右值引用? 「腾讯」
62. 【其他算法｜L3】流数据系统，输出元组递归 「百度」
63. 【其他算法｜L3】介绍一下禁忌搜索。 「京东」
64. 【其他算法｜L3】可选:KMP/马拉车/扫描线算法/蓄水池算法/flood fill 等 「阿里巴巴」
65. 【其他算法｜L3】介绍一下粒子群优化。 「京东」
66. 【其他算法｜L3】基类指针如何找到实际对象的虚表? 「腾讯」
67. 【并发、锁｜L3】基于互斥锁实现读写锁 「腾讯混元」
68. 【配置管理、Mock｜L3】实现配置管理模块，后端模型服务可使用Mock。 「蚂蚁集团」
69. 【模拟｜L3】实现tournament，根据比赛流程，最后打印一个表格出来。 「字节Seed」
70. 【并发、计数器｜L3】实现一个支持高并发的计数器(读写锁/LongAdder/CAS自旋) 「阿里通义」
71. 【[LC 146](https://leetcode.cn/problems/lru-cache/)/[LC 460](https://leetcode.cn/problems/lfu-cache/)｜设计、哈希表、链表、双向链表｜L4】Coding: LRU Cache。LRU Cache的核心思路。手写LRU/LFU缓存算法，用于大模型对话缓存、高频Query缓存优化场景。；另一场次补充：实现LRU Cache：1. 要求get/put均为O(1)。2. 带过期时间的LRU。；另一场次补充：手撕题：实现LRU（逻辑存储）；容易写错哪些地方？不变量是什么？；另一场次补充：手写LRUCache。 「MiniMax、京东、字节跳动、腾讯、阿里通义」
72. 【推荐系统、SIM、性能分析｜L4】线上SIM的实现逻辑，耗时怎么看？ 「阿里巴巴」
73. 【综合编程｜L4】做一个前端的AI助手页面。 「阿里巴巴」
74. 【设计、综合编程｜L4】比如一段用协程实现的数据加载方案，需要你发现生命周期管理有没有隐患？ 「阿里巴巴」
75. 【[LC 146](https://leetcode.cn/problems/lru-cache/)｜设计、哈希表、链表、双向链表｜L4】写一个多线程的简单题（好像是RWLock，记不清了）；写一个LRU缓存；写一个迷之日志系统，场景题。 「字节跳动」
76. 【推理加速、访存分析｜L4】手算DeepSeek V3每次推理的访存量 「字节跳动」
77. 【综合编程｜L4】怎么做路径平滑，目标函数设置、平滑度计算公式、约束条件？ 「百度」
78. 【设计｜L4】用户会反复调整K；不能每次调整都让用户等很久，要优化latency。预处理时就把每个标点的位置和间隔文本长度一起存下来；后面长度直接查、不用重新遍历。 「阿里巴巴」
79. 【设计、模拟、哈希表、滑动窗口｜L5】请设计一个算法高效地从海量交易流水中检测异常模式(如欺诈交易)。简述你的思路、用到的核心算法和复杂度分析。 「蚂蚁集团」

## AI手撕题

**Transformer与Attention**

1. 【Transformer、伪代码｜L2】写出伪代码 用abc 预测第四个token 告诉了我特征维度 头个数。 「京东」
2. 【Transformer、Attention、手撕｜L3】手撕Cross Attention 「百度」
3. 【cross attention、模型实现｜L3】代码：MCG模型，类似于crossattention。 「百度」
4. 【Transformer、Attention、MQA｜L3】实现MQA。 「腾讯」
5. 【Transformer、Attention、手写｜L3】手写论文中的堆叠的self-attention和cross-attention层，解释一下有什么作用？ 「京东」
6. 【Transformer、Attention、MHA｜L3】自注意力，写完让直接在代码上改成mha; 「阿里通义」
7. 【Transformer、Attention、Causal Mask｜L4】手撕:带因果掩码的缩放点积注意力前向传播代码 「字节跳动」
8. 【Transformer、MHA、Causal Mask、复杂度｜L4】实现Causal mask的MHA，并说明计算复杂度 「美团」
9. 【Transformer、Decoder、深度学习｜L4】手撕decoder layer。 「阿里巴巴」
10. 【Transformer、GQA、手撕、Attention、注意力机制、深度学习｜L4】写一下GQA（Grouped Query Attention）的实现 「京东、字节跳动、百度、阿里巴巴」
11. 【Transformer、Attention、Linear Attention｜L4】手撕linear attention 「字节跳动」
12. 【Transformer、Attention、MHA、MQA、GQA｜L4】手撕:mha mqa gqa的兼容版 init里设置传参通义万相数据。 「阿里通义」
13. 【Transformer、Attention、MQA、GQA｜L4】手撕MQA和GQA 「阿里巴巴」
14. 【Transformer、Attention、RoPE｜L4】手撕MHA + RoPE。 「字节跳动」
15. 【Transformer、Attention、MHA、手撕、NumPy、Mask、Self-Attention、Token Mixing、contiguous、Positional Encoding｜L4】Attention机制包含哪些核心算子？请用PyTorch简单搭建一个Self-Attention模块的伪代码，并解释每个算子的作用。；另一场次补充：手写自注意力，然后追问在Attention is all you need一文中，除了Q/K/V的注意力计算，还有一个关键步骤是什么？；另一场次补充：手写Multi-Head Self-Attention的实现(PyTorch或TensorFlow)；另一场次补充：大模型手撕:多头注意力机制，softmax(以及softmax处理数值溢出的做法)；另一场次补充：手撕Multi-Head Attention contiguousO的作用。；另一场次补充：实现多头自注意力（Multi-Head Attention）机制的核心代码。；另一场次补充：手撕两道，第一道是自注意力机制，我直接写了mha，然后就不用写第二道了。；另一场次补充：手写attention，解释scale为什么除以√dk而不是dk。；另一场次补充：手写Multi-Head Attention（带Mask）；另一场次补充：multi-head attention具体怎么实现？；另一场次补充：手撕多头注意力。但是这部分不会考很难，也不用太担心。；另一场次补充：请实现Multi-Head Attention。；另一场次补充：Pytorch实现MHA和打印螺旋矩阵 2选1；另一场次补充：手撕selfattention(用矩阵乘法)。；另一场次补充：手撕multi-head attention。；另一场次补充：手撕multi-head attention；另一场次补充：手撕代码：self-attention；另一场次补充：手撕MHAtokenmixing；另一场次补充：Attention算子的实现。；另一场次补充：手撕多头注意力机制（MHA）。；另一场次补充：写出attention公式。；另一场次补充：手撕：用numpy，写MHA；另一场次补充：手撕MHA（带mask）；另一场次补充：手撕多头自注意力机制。；另一场次补充：代码：实现多头自注意力；另一场次补充：手撕题，例如手写MHA；另一场次补充：手撕位置编码和MHA；另一场次补充：手撕:多头注意力机制；另一场次补充：实现多头注意力机制；另一场次补充：手撕MHA和场景题；另一场次补充：实现多头自注意力。；另一场次补充：手撕自注意力机制；另一场次补充：实现多头自注意力；另一场次补充：手撕多头注意力。；另一场次补充：手撕：MHA。；另一场次补充：手撕MHA代码；另一场次补充：手撕多头注意力；另一场次补充：手撕是mha。；另一场次补充：手撕自注意力。；另一场次补充：手撕MHA。；另一场次补充：手撕MHA 「京东、字节Seed、字节跳动、拼多多、百度、美团、腾讯、腾讯混元、蚂蚁集团、阿里巴巴、阿里通义」
16. 【Transformer、MLA、手撕、Attention｜L4】手撕MLA。 「京东、字节跳动」
17. 【Transformer、Position Embedding、RoPE、位置编码｜L4】实现Rotary Position Embedding (RoPE)；另一场次补充：手写RoPE旋转位置编码 「字节跳动、美团」
18. 【MHA、KV Cache｜L4】手写带kvcache的MHA。 「百度」
19. 【Transformer、Attention、MHA、MGA｜L4】手撕:实现多头注意力模块(MHA)，并要优化为MGA 「字节Seed」
20. 【Transformer、Decoder｜L4】手撕:实现一个transformer decoder 「字节Seed」
21. 【Transformer、Attention、优化｜L4】设计高效Attention实现，降低时间/空间复杂度 「字节跳动」
22. 【Transformer、Encoder、实现、手撕｜L5】手撕Transformer和一道top100；另一场次补充：手撕Transformers encoder；另一场次补充：实现Transformer encoder 「拼多多、百度、美团、阿里通义」

**模型结构与基础组件**

1. 【MLP、PyTorch、图像｜L2】手写MLP，用torch，实现很简单，然后追问，如果要把这个MLP用在图像上，你怎么修改? 「字节跳动」
2. 【MLP｜L2】手撕mlp。 「字节跳动」
3. 【PyTorch、神经网络｜L2】手撕:对着屏幕用pytorch实现一个神经网络 「蚂蚁集团」
4. 【ViT、参数量计算｜L2】手撕:写一个计算ViT参数量的函数，自己定义输入(层数、纬度等)输出模型总参数量。 「字节Seed」
5. 【Mixup、Affine、RankMixer｜L3】手撕简单写了 mixup和 affn，还是rankmixer的内容。 「美团」
6. 【二分类、神经网络｜L3】code 写一个二分类网络 「美团」
7. 【神经网络、多分类、训练｜L3】手写神经网络实现多分类训练。 「MiniMax」
8. 【大语言模型、伪代码、模型架构｜L4】手撕DeepSeek V3的结构伪代码 「字节跳动」
9. 【Transformer、Attention、LoRA、MHA｜L4】实现一个MultiHeadAttention，要求里面的Q、K、V的线性变换层引入LORA技术。 「腾讯」
10. 【生成模型、VQ-VAE、伪代码｜L4】手撕VQVAE伪码。 「阿里巴巴」
11. 【NumPy、前馈网络、全连接、激活函数、损失函数｜L4】用np实现一个神经网络的前馈层，四层全连接，2-3个激活函数，loss函数 「美团」
12. 【ResNet、PyTorch｜L4】手撕torch系列：ResNet(深度残差网络)原理及代码实现(基于Pytorch)。 「腾讯」

**激活、归一化与损失函数**

1. 【损失函数、Hinge Loss｜L1】讲一下hinge loss，写出公式 「蚂蚁集团」
2. 【数学、激活函数、公式｜L1】sigmoid的代码手写。；另一场次补充：写一下sigmoid的公式 「百度、阿里巴巴」
3. 【L1、L2、正则化｜L1】介绍一下L1和L2正则化，写一下公式。 「阿里巴巴」
4. 【损失函数、逻辑回归｜L1】写一下逻辑回归的损失函数 「拼多多」
5. 【损失函数｜L2】手撕4个损失函数 「腾讯」
6. 【损失函数、交叉熵｜L2】写出交叉熵公式并推导。 「百度」
7. 【损失函数、交叉熵｜L2】写出交叉熵损失函数公式。；另一场次补充：交叉熵损失函数写一下；另一场次补充：交叉熵的代码手写。 「字节跳动、腾讯混元、阿里巴巴」
8. 【损失函数、Python｜L3】10个常用的损失函数及Python代码实现。 「腾讯」
9. 【BatchNorm、归一化｜L3】手撕:batch累积的均值和方差(类似batchnorm) 「字节跳动」
10. 【深度学习基础、归一化与激活｜L3】写一下DeepNorm代码实现 「腾讯混元」
11. 【BCE、InfoNCE、损失函数、Cross Entropy、对比学习、CLIP｜L3】手撕: infoNCE loss Cross Entropy 怎么算，为什么要normalize，温度系数的作用；另一场次补充：实现InfoNCE Loss（CLIP）。；另一场次补充：手撕BCE,InfoNCE损失。；另一场次补充：写下 infoNCE loss？ 「阿里巴巴、阿里通义」
12. 【生成模型、损失函数、VAE｜L3】手撕VAE训练loss。 「腾讯混元」
13. 【RMSNorm、层归一化、LayerNorm、归一化、深度学习基础、归一化与激活｜L3】实现RMSNorm并解释/实现LayerNorm并解释/实现BatchNorm并解释/实现Decoder block；另一场次补充：手撕RMSnorm/layerNorm/BatchNorm；另一场次补充：手撕layernormalization。；另一场次补充：写一个layer norm；另一场次补充：手写RMSNorm层；另一场次补充：手写RMSNorm。 「字节跳动、百度、美团、阿里巴巴」
14. 【SwiGLU、激活函数｜L3】手写SwiGLU。 「字节跳动、美团」
15. 【softmax、online、Softmax、C++、模板、手撕、实现、深度学习基础、公式推导、数值稳定性｜L3】介绍Softmax函数的使用及实现。；另一场次补充：实现Softmax的CPU C++实现，模版；另一场次补充：手撕题:online softmax；另一场次补充：手撕softmax 函数怎么写？；另一场次补充：手撕online softmax；另一场次补充：手撕safe softmax；另一场次补充：写出softmax公式。 「字节跳动、百度、腾讯、蚂蚁集团、阿里巴巴」
16. 【堆（优先队列）、分治、快速选择｜L4】上来就写题：1. LeetCode找TopK（即求数组中第K大或前K大元素）；2. 任一深度学习框架实现BatchNorm（Batch Normalization）的前向和反向传播。 「阿里巴巴」

**训练优化与自动微分**

1. 【Adam、公式｜L1】写出adam计算公式。 「字节跳动」
2. 【AdamW、优化器｜L3】手撕AdamW。 「腾讯」
3. 【综合编程、神经网络、前向传播、反向传播、梯度｜L3】手撕代码第一道：forward与backward；第二道：price股票价格那道题/leetcode原题。；另一场次补充：手撕:两层神经网络，要求自己定义参数矩阵实现前向传播和反向传播 (自己求梯度)；另一场次补充：手撕两层神经网络，forward backward propagation。 「字节Seed、百度文心、阿里巴巴」
4. 【学习率调度、动态｜L3】手写实现动态学习率调度 「字节跳动」
5. 【反向传播、矩阵求导｜L3】code: 求偏导 y=f(C); C=AB; A输入、B参数、f激活函数、y输出，已知y对C的偏导是P，求y对B的偏导矩阵大小: A[m,k]; B[k,n]; C[m,n]; P[m,n] 「阿里巴巴」
6. 【深度学习基础、训练稳定性｜L3】手写知识蒸馏 「字节跳动」
7. 【梯度下降、线性回归、训练流程｜L3】基于梯度下降实现线性回归完整训练流程，要求运行，loss收敛。 「MiniMax」
8. 【Adam、优化器、偏差校正｜L4】实现MyAdam优化器：MyAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)。方法：step()使用偏差校正后的矩更新参数，zero_grad()将所有参数梯度清零。约束：必须与torch.optim.Adam数值一致，偏差校正：m_hat=m/(1-beta1^t)。；另一场次补充：从零实现Adam优化器。 「字节跳动」
9. 【反向传播、手动实现｜L4】手动实现反向传播。 「腾讯」
10. 【LoRA、微调｜L4】手撕 LoRA，现场发邮箱题目链接，和面试是两个页面。 「阿里通义」

**强化学习与对齐**

1. 【强化学习、DPO、损失函数｜L3】实现DPO损失函数：dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1) -> Tensor。参数：policy_chosen_logps策略模型对选中回复的对数概率(B,)，policy_rejected_logps策略模型对拒绝回复的对数概率(B,)，ref_chosen_logps, ref_rejected_logps参考模型的对数概率(B,)，beta温度缩放因子。返回：标量损失。约束：L=-log(sigmoid(beta * ((pi_c-ref_c)-(pi_r-ref_r)))).mean()。；另一场次补充：dpo训练的损失函数和训练目标，dpo如何改进，dpo的loss实现；另一场次补充：实现dpo损失，按照想法能写多少是多少，不要求跑通。；另一场次补充：实现DPO(直接偏好优化)损失。；另一场次补充：写一下DPO loss函数。；另一场次补充：手撕DPO的损失函数。；另一场次补充：手撕DPO公式loss 「字节Seed、字节跳动、百度、腾讯、蚂蚁集团」
2. 【强化学习、DQN｜L3】写一下DQN的伪代码 「字节跳动」
3. 【RLHF、PPO、DPO、损失函数｜L3】讲一下RLHF的流程，写一下PPO和DPO的LOSS表达式。 「阿里通义」
4. 【PPO、强化学习、损失函数｜L3】PPO实现细节，核心思想，损失函数。 「字节跳动」
5. 【强化学习、RLHF、损失函数｜L3】在RLHF中，目前主流的强化学习算法有哪几个，写一下损失函数的表达式。；另一场次补充：在RLHF中，目前主流的强化学习算法有哪些，写一下损失函数的表达式？ 「腾讯、腾讯混元」
6. 【GRPO、强化学习、代码实现｜L4】如果落到代码实现，流程原理是什么? 「字节跳动」
7. 【强化学习、GRPO、损失函数｜L4】手撕:GRPO loss，讲各个tensor的shape 「字节跳动」
8. 【GRPO、强化学习、KL散度、优势函数、损失｜L4】共享屏幕写出来GRPO的每个计算步骤(大致公式)，并解释其中kl散度怎么计算的，adv函数怎么计算，token级别的损失怎么计算的。 「京东」
9. 【PPO、DPO、强化学习｜L4】手撕PPO和DPO。 「字节Seed」
10. 【强化学习、PPO｜L4】实现PPO。 「百度、腾讯」
11. 【强化学习、奖励模型、策略梯度｜L4】手写奖励模型与策略梯度更新 「字节跳动」
12. 【[LC 437](https://leetcode.cn/problems/path-sum-iii/)｜树、深度优先搜索、二叉树｜L4】三个code: 1. 智力题（具体题目未给出）；2. 二叉树中路径和为target的路径数量（路径从根节点到叶子节点？或任意节点到任意节点？需明确）；3. GRPO伪代码。 「阿里通义」
13. 【强化学习、重要性采样｜L4】手撕: sequence importance sampling和token importance sampling 「百度」
14. 【DAPO、强化学习、损失函数、clip、奖励函数、优势函数、Soft IoU｜L5】手撕 DAPO 整个过程，包括奖励函数、组内均值、优势函数以及 loss，外部套一层动态采样，我的论文 Soft IoU 是怎么算的也写进来。；另一场次补充：手撕DAPO，没写出来，我说了下loss计算，然后让我打开论文对着公式详细讲一下，并提问clip的作用。 「京东、字节跳动」
15. 【GRPO、强化学习、代码实现｜L5】如果给一个真实query,让你代码实现GRPO全流程,你会如何组织 rollout、reward、logprob、advantage、loss 和 update? 「字节跳动」

**机器学习与评估指标**

1. 【数学、数组｜L2】手写实现余弦相似度计算，适配Embedding向量检索场景（一维向量输入）。 「阿里通义」
2. 【推荐指标、GAUC｜L3】手撕代码：Request维度的GAUC计算；另一场次补充：Gauc的计算，写一下 「字节跳动、美团、阿里巴巴」
3. 【数学｜L3】手写Kmeans 「阿里巴巴」
4. 【随机化、设计｜L3】手撕bucket sampler。 「腾讯混元」
5. 【AUC、模型评估、评估指标、Python｜L4】手撕：给了label和预测score，实现AUC的代码，时间复杂度-优化。；另一场次补充：手撕一下AUC(不用numpy)。；另一场次补充：用Python实现AUC的计算。；另一场次补充：手撕AUC/常见损失函数；另一场次补充：代码：手撕AUC。 「京东、字节Seed、字节跳动、百度、阿里巴巴」

**推荐搜索与数据算法**

1. 【推荐系统、Cross-Encoder、粗排｜L3】粗排: Cross-Encoder对top-100重打分(比Bi-Encoder准但慢) 「百度」
2. 【推荐系统、FM、实现｜L3】实现FM（Factorization Machine）模型 「字节跳动」
3. 【推荐模型、FM｜L3】实现FM（因子分解机） 「字节跳动」
4. 【推荐模型、DIN｜L4】实现DIN（Deep Interest Network） 「阿里巴巴」
5. 【向量检索、FAISS、实现｜L4】实现FAISS（近似最近邻搜索库） 「字节跳动」
6. 【推荐模型、MMoE｜L4】实现MMoE（Multi-gate Mixture-of-Experts） 「百度、阿里巴巴」
7. 【堆（优先队列）、分治、排序｜L4】Top-K问题：从1亿个数中找出前10大的数；手写Top-K检索算法，实现向量库快速筛选，适配RAG召回场景；分片聚合，再做二阶段Top-K。 「字节跳动、腾讯、阿里通义」
8. 【推荐系统、多目标排序、MMOE、PLE｜L4】多目标排序的具体实现 share-nothing、share-bottom、MMOE、PLE？ 「腾讯」
9. 【推荐模型、DIN、时间衰减｜L4】实现一个time_decay的DIN（Deep Interest Network with time decay） 「阿里巴巴」
10. 【推荐系统、条件概率｜L4】给定用户历史行为序列，根据历史行为序列算出元组对的条件概率，并根据条件概率进行用户推荐。 「腾讯」
11. 【推荐系统、哈希检索、target attention｜L4】长序列建模 ETA / SDIM / TWIN: 哈希检索 + target attention 的工程实现 「字节跳动」

**多模态与计算机视觉**

1. 【[LC 223](https://leetcode.cn/problems/rectangle-area/)｜几何、数学｜L2】手撕矩形面积（就是IoU计算）。 「腾讯混元」
2. 【插值、图像处理｜L2】手撕插值算法。 「拼多多」
3. 【IoU、损失函数、目标检测｜L3】手写IoU损失，要求跑通。 「百度」
4. 【NMS、目标检测、计算机视觉｜L3】手撕NMS(非极大值抑制)代码，要求一次写对。；另一场次补充：手撕NMS过程。 「字节跳动、拼多多、阿里巴巴」
5. 【ViT、损失函数、PyTorch｜L3】手撕 import torch import torch.nn as nn class ViTCrossEntropyLoss(nn.Module): 「百度」
6. 【ViT、损失函数｜L3】手撕: ViT loss(图。 「百度」
7. 【BEV、网格采样、多视角融合｜L4】给你一个BEV特征图，手写一个Grid Sampler(网格采样器)，把周围视角的特征投影过来。 「字节跳动」
8. 【图像处理、Attention、ViT｜L4】手撕从图片处理到attention的具体代码 「阿里巴巴」
9. 【数学、几何｜L4】斜框IOU实现。计算两个旋转矩形（斜框）的交并比（IOU）。 「腾讯」

**CUDA、Triton与高性能算子**

1. 【CUDA、向量加法、优化｜L2】写一个vector add的CUDA kernel核心内容。；另一场次补充：CUDA Kernel:实现向量加法并解释优化思路 「百度」
2. 【[LC 26](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)｜数组、双指针｜L2】手撕代码，leetcode 26（删除有序数组中的重复项），CUDA矩阵乘。 「百度」
3. 【CUDA、reduce｜L3】手撕CUDA代码二维矩阵，每一行做 reduce。 「百度」
4. 【CUDA、前缀和｜L3】写cuda算子：前缀和。实现一个base的，讲优化方法。 「美团」
5. 【量化、反量化、INT8｜L3】实现INT8 Per-Tensor量化与反量化函数 「腾讯」
6. 【卷积、滑窗｜L3】手写2D卷积的实现过程(不能用API，要写滑窗逻辑)。 「字节跳动」
7. 【模型并行、通信原语｜L3】手写算法:实现一个简单的模型并行通信原语。 「腾讯」
8. 【量化、伪量化、训练校准｜L4】设计8bit量化算法，手写权重训练校准与伪量化过程 「字节跳动」
9. 【CUDA、GEMM｜L4】写cuda算子gemm：实现一个base的，讲优化方法。 「美团」
10. 【CUDA、reduce、优化、warp shuffle｜L4】写一个reduce，用block，优化版:用warp shuffle，能不能再优化?；另一场次补充：手撕reduce（至少要用warp shuffle）；另一场次补充：手撕reduce和reduce的优化 「字节跳动、阿里巴巴」
11. 【CUDA、Triton、归一化、算子融合｜L4】代码题是CUDA or Triton实现一个fuse的norm算子 「字节跳动」
12. 【CUDA、GEMM、优化｜L4】手撕GEMM的一些优化方法（一个thread算多个数，数据复用；访存合并；双buffer边读边算；避免bank conflict） 「字节跳动」
13. 【CUDA、矩阵乘法、分块｜L4】写代码题：tiled matmul。 「字节Seed」
14. 【CUDA、reduce、二维矩阵｜L4】手撕二维大矩阵reduce，(1000000,128)矩阵reduce到(1,128) 「字节跳动」
15. 【分布式训练、梯度分片、环式通信｜L4】手写实现：在多GPU环境下，使用梯度分片与环式通信优化传输效率。 「字节跳动」
16. 【卷积、IM2COL、GEMM｜L4】卷积的实现IM2COL+GEMM。 「百度」
17. 【CUDA、Triton、量化、GEMM｜L5】用CUDA/Triton/Tilelang/任何我可以的语言实现一个quant GEMM算子 「腾讯混元」
18. 【Triton、Attention、online softmax、tiling｜L5】用Triton实现一个tiling加online softmax的Attention kernel，在SRAM上完成局部计算避免中间矩阵写入HBM，对比标准PyTorch实现，在seq_len等于2048的设定下HBM读写量减少了百分之六十。 「阿里巴巴」

**AI系统与综合实现**

1. 【数学｜L2】可以用numpy写一下公式吗？ 「美团」
2. 【参数量计算、DeepSeek V3｜L3】手撕DeepSeek V3的参数量手算 「字节跳动」
3. 【调度算法、Gang Scheduling｜L3】实现一个简单的Gang Scheduler调度算法。 「腾讯」
4. 【大模型、推理优化、KVCache｜L3】手写KVCache的实现逻辑 「百度」
5. 【大语言模型、PPL、mask、代码纠错｜L3】手撕：PPL计算以及mask的代码哪里有问题？ 「字节跳动」
6. 【采样、解码策略、大语言模型｜L3】手写解码策略中的Top-k Sampling或Top-p Sampling 「百度」
7. 【大语言模型、采样、top-k、归一化｜L3】手撕：实现一个最简单的 top-k采样（给定 logits/概率，取top-k后重新归一化采样），并说明边界情况怎么处理。 「拼多多」
8. 【双指针、字符串｜L3】手撕:删除字符串中的重复元素II aicoding(搭建一个推荐页面) 「美团」
9. 【Tokenizer、PyTorch｜L3】实现一个Tokenizer，只能用PyTorch基础语法。 「百度、百度文心」
10. 【综合编程｜L4】写一下DeepResearch伪代码（可能指类似Deep Research的搜索或推理流程）。 「蚂蚁集团」
11. 【HTTP网关、负载均衡、KVCache、大模型推理｜L4】实现一个面向大模型推理场景的HTTP网关，支持：智能路由、负载均衡、KVCache感知调度。 「蚂蚁集团」
12. 【MoE、混合专家模型、并行处理｜L4】手撕：moe并行处理 「字节跳动、阿里通义」
13. 【综合编程｜L4】代码环节是让我打开一个vibecoding界面，制作一个agent，agent具体功能他来提供。 「阿里巴巴」
14. 【vllm、推理加速｜L4】手撕：vllm 「阿里通义」
15. 【IOU、NMS、卷积、池化、BN、ResNet、FCN、多头注意力｜L4】手撕IOU、NMS、正向卷积、池化、BN、resnet、FCN、多头注意力。 「腾讯」
16. 【流水线并行、计算通信重叠｜L4】手写流水线并行策略，将模型分层部署到多设备，计算通信重叠 「字节跳动」
17. 【模型设计、生成理解统一｜L4】手写一个输出端生成理解统一的模型，你会怎么设计？ 「腾讯」
18. 【KV Cache、Transformer、推理优化｜L4】实现推理时的键值缓存机制，管理与优化长文本生成 「字节跳动」
