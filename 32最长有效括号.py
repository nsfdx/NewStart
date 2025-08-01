class DetailedSolution:
    def longestValidParenthesesDP(self, s: str) -> int:
        """
        方法1: 动态规划详细解释
        """
        if not s:
            return 0
        
        n = len(s)
        # dp[i] 表示以下标 i 字符结尾的最长有效括号的长度
        dp = [0] * n
        max_len = 0
        
        print(f"输入字符串: {s}")
        print(f"初始dp数组: {dp}")
        print("=" * 50)
        
        for i in range(1, n):
            print(f"\n处理位置 {i}: s[{i}] = '{s[i]}'")
            
            # 只有当前字符是 ')' 时才可能形成有效括号
            if s[i] == ')':
                # 情况1: 形如 ...() 的情况
                if s[i-1] == '(':
                    # 前面是 '('，可以直接配对
                    # dp[i] = dp[i-2] + 2 (如果 i-2 >= 0)
                    dp[i] = (dp[i-2] if i >= 2 else 0) + 2
                    print(f"  情况1: s[{i-1}]='(' -> 直接配对")
                    print(f"  dp[{i}] = {dp[i-2] if i >= 2 else 0} + 2 = {dp[i]}")
                
                # 情况2: 形如 ...)) 的情况，且前一个位置已经有有效括号
                elif dp[i-1] > 0:
                    # 需要找到与当前 ')' 匹配的 '(' 的位置
                    # 跳过前面的有效括号序列
                    match_pos = i - dp[i-1] - 1
                    print(f"  情况2: s[{i-1}]=')' 且 dp[{i-1}]={dp[i-1]}")
                    print(f"  寻找匹配位置: {i} - {dp[i-1]} - 1 = {match_pos}")
                    
                    if match_pos >= 0 and s[match_pos] == '(':
                        # 找到了匹配的 '('
                        # dp[i] = dp[i-1] + 2 + dp[match_pos-1]
                        before_match = dp[match_pos-1] if match_pos > 0 else 0
                        dp[i] = dp[i-1] + 2 + before_match
                        print(f"  找到匹配: s[{match_pos}]='('")
                        print(f"  dp[{i}] = {dp[i-1]} + 2 + {before_match} = {dp[i]}")
                    else:
                        print(f"  未找到匹配的'(' -> dp[{i}] = 0")
                else:
                    print(f"  前一个字符是')' 但dp[{i-1}]=0 -> dp[{i}] = 0")
            else:
                print(f"  当前字符是'(' -> dp[{i}] = 0")
            
            print(f"  当前dp数组: {dp}")
            max_len = max(max_len, dp[i])
        
        print(f"\n最终结果: {max_len}")
        return max_len
    
    def longestValidParenthesesStack(self, s: str) -> int:
        """
        方法2: 栈方法详细解释
        """
        if not s:
            return 0
        
        # 栈用来存储下标
        # 初始时放入 -1 作为基准点
        stack = [-1]
        max_len = 0
        
        print(f"输入字符串: {s}")
        print(f"初始栈状态: {stack}")
        print("=" * 50)
        
        for i, char in enumerate(s):
            print(f"\n处理位置 {i}: s[{i}] = '{char}'")
            print(f"处理前栈状态: {stack}")
            
            if char == '(':
                # 遇到 '('，将其下标入栈
                stack.append(i)
                print(f"  '(' 入栈 -> 栈状态: {stack}")
            
            else:  # char == ')'
                # 遇到 ')'，弹出栈顶元素
                popped = stack.pop()
                print(f"  ')' 弹出元素: {popped} -> 栈状态: {stack}")
                
                if not stack:
                    # 栈为空，说明当前 ')' 没有匹配的 '('
                    # 将当前下标作为新的基准点入栈
                    stack.append(i)
                    print(f"  栈为空，当前')' 无匹配 -> 将{i}作为新基准点入栈: {stack}")
                else:
                    # 栈不为空，计算当前有效括号长度
                    # 长度 = 当前位置 - 栈顶位置
                    current_len = i - stack[-1]
                    max_len = max(max_len, current_len)
                    print(f"  有效配对！长度 = {i} - {stack[-1]} = {current_len}")
                    print(f"  当前最大长度: {max_len}")
        
        print(f"\n最终结果: {max_len}")
        return max_len

def demonstrate_with_examples():
    """
    用具体例子演示两种方法
    """
    solution = DetailedSolution()
    
    examples = [
        "()(()",      # 简单例子
        ")()())",     # 复杂例子
        "(()())",     # 嵌套例子
    ]
    
    for example in examples:
        print("\n" + "="*80)
        print(f"示例: '{example}'")
        print("="*80)
        
        print("\n【动态规划方法】")
        print("-" * 40)
        result_dp = solution.longestValidParenthesesDP(example)
        
        print("\n【栈方法】")
        print("-" * 40)
        result_stack = solution.longestValidParenthesesStack(example)
        
        print(f"\n两种方法结果一致: {result_dp == result_stack}")

def explain_key_concepts():
    """
    解释关键概念
    """
    print("\n" + "="*80)
    print("关键概念解释")
    print("="*80)
    
    print("""
【动态规划方法核心思想】
1. dp[i] 定义：以字符 s[i] 结尾的最长有效括号子串长度
2. 只有 s[i] = ')' 时，dp[i] 才可能 > 0
3. 两种转移情况：
   - s[i-1] = '('：直接配对，dp[i] = dp[i-2] + 2
   - s[i-1] = ')' 且 dp[i-1] > 0：找到匹配的 '('，连接前后的有效序列

【栈方法核心思想】
1. 栈存储字符的下标，不是字符本身
2. 栈底始终保持一个"基准点"（最后一个未匹配的 ')' 的位置）
3. 遇到 '(' 就入栈；遇到 ')' 就出栈，然后计算长度
4. 如果出栈后栈为空，说明当前 ')' 无匹配，将其作为新基准点

【为什么这两种方法有效】
- 动态规划：通过状态转移确保每个位置的最优解
- 栈方法：通过维护基准点，能够正确计算任意有效括号序列的长度
    """)

if __name__ == "__main__":
    demonstrate_with_examples()
    explain_key_concepts()