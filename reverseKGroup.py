import collections
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        que = collections.deque()
        idx = head
        left,right,nextNode = None,None,None
        curCnt = 0
        dequeCnt = 0
        iteration_count = 0  # 添加计数器来检测无限循环

        while idx:
            iteration_count += 1
            if iteration_count > 20:  # 防止真的无限循环
                print(f"检测到无限循环！已执行 {iteration_count} 次迭代")
                print(f"当前 idx.val: {idx.val}")
                print(f"当前 curCnt: {curCnt}")
                print(f"que 长度: {len(que)}")
                break
                
            if curCnt == k:
                print(f"开始处理第 {dequeCnt + 1} 组，curCnt={curCnt}")
                while que:
                    right = que.pop()
                    print(f"弹出节点: {right.val}, 剩余队列长度: {len(que)}")
                    if curCnt == k:
                        nextNode = right.next
                        print(f"设置 nextNode: {nextNode.val if nextNode else None}")
                    if left:
                        left.next = right
                    elif not dequeCnt:
                        head = right
                    left = right
                    curCnt-=1
                dequeCnt+=1
                idx = nextNode
                print(f"更新 idx 到: {idx.val if idx else None}")
            else:
                que.append(idx)
                curCnt+=1
                idx = idx.next
                print(f"添加节点: {que[-1].val}, curCnt={curCnt}")
        if curCnt:
            left.next = que.popleft()
        return head

def create_linked_list(vals):
    if not vals:
        return None
    head = ListNode(vals[0])
    current = head
    for val in vals[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

def print_linked_list(head, max_nodes=10):
    result = []
    current = head
    count = 0
    while current and count < max_nodes:
        result.append(current.val)
        current = current.next
        count += 1
    if current:
        result.append("...")
    return result

# 测试用例 - 这将证明无限循环
if __name__ == "__main__":
    print("=== 测试无限循环问题 ===")
    solution = Solution()
    
    # 创建测试链表: [1, 2, 3, 4, 5]
    head = create_linked_list([1, 2, 3, 4, 5])
    print("原始链表: [1, 2, 3, 4, 5]")
    print("k = 2")
    print("开始执行...")
    
    try:
        result = solution.reverseKGroup(head, 2)
        print("结果:", print_linked_list(result))
    except Exception as e:
        print(f"发生异常: {e}")