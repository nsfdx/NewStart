def longest_palindrome(s: str) -> str:
    """
    查找字符串中的最长回文子串
    使用中心扩展法
    """
    if not s:
        return ""
    
    start, end = 0, 0
    
    for i in range(len(s)):
        # 检查奇数长度的回文（以i为中心）
        len1 = expand_around_center(s, i, i)
        # 检查偶数长度的回文（以i和i+1为中心）
        len2 = expand_around_center(s, i, i + 1)
        
        max_len = max(len1, len2)
        
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    
    return s[start:end + 1]


def expand_around_center(s: str, left: int, right: int) -> int:
    """
    从中心向两边扩展，找到最长回文的长度
    """
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return right - left - 1


def longest_palindromic_subsequence(s: str) -> str:
    """
    查找字符串中的最长回文子序列（使用动态规划）
    注意：这与最长回文子串不同，子序列不要求连续
    """
    n = len(s)
    if n == 0:
        return ""
    
    # dp[i][j] 表示 s[i:j+1] 中最长回文子序列的长度
    dp = [[0] * n for _ in range(n)]
    
    # 单个字符都是回文
    for i in range(n):
        dp[i][i] = 1
    
    # 填充dp表
    for length in range(2, n + 1):  # 子串长度
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2:
                    dp[i][j] = 2
                else:
                    dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    
    # 重构最长回文子序列
    def construct_palindrome(i, j):
        if i > j:
            return ""
        if i == j:
            return s[i]
        
        if s[i] == s[j]:
            return s[i] + construct_palindrome(i + 1, j - 1) + s[j]
        elif dp[i + 1][j] > dp[i][j - 1]:
            return construct_palindrome(i + 1, j)
        else:
            return construct_palindrome(i, j - 1)
    
    return construct_palindrome(0, n - 1)


# 测试函数
if __name__ == "__main__":
    test_strings = [
        "babad",
        "cbbd", 
        "racecar",
        "abcdef",
        "character",
        "aabbaa"
    ]
    
    print("=== 最长回文子串测试 ===")
    for s in test_strings:
        result = longest_palindrome(s)
        print(f"字符串: '{s}' -> 最长回文子串: '{result}'")
    
    print("\n=== 最长回文子序列测试 ===")
    for s in test_strings:
        result = longest_palindromic_subsequence(s)
        print(f"字符串: '{s}' -> 最长回文子序列: '{result}'")