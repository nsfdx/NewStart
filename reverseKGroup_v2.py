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
        left = None
        newHead = None
        
        while idx:
            # 先保存下一个节点，避免链表断裂后找不到
            nextNode = idx
            
            # 收集k个节点
            for _ in range(k):
                if not nextNode:
                    # 不足k个节点，直接连接剩余部分
                    if left:
                        left.next = idx
                    return newHead if newHead else head
                que.append(nextNode)
                nextNode = nextNode.next
            
            # 反转这k个节点
            prev = nextNode  # 下一组的开始或者None
            while que:
                current = que.pop()  # 从右侧弹出实现反转
                current.next = prev
                prev = current
            
            # 连接反转后的组
            if left:
                left.next = prev  # 连接到上一组
            else:
                newHead = prev    # 第一组，设置新头节点
            
            # 找到当前组的尾节点（原来的第一个节点）
            while left and left.next != nextNode:
                left = left.next
            if not left:
                left = que[0] if que else prev
                while left.next != nextNode:
                    left = left.next
            
            # 继续下一组
            idx = nextNode
        
        return newHead if newHead else head

# 测试代码
def create_linked_list(vals):
    if not vals:
        return None
    head = ListNode(vals[0])
    current = head
    for val in vals[1:]:
        current.next = ListNode(val)
        current = current.next
    return head

def print_linked_list(head):
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result

if __name__ == "__main__":
    solution = Solution()
    
    # 测试
    head = create_linked_list([1, 2, 3, 4, 5])
    result = solution.reverseKGroup(head, 2)
    print(print_linked_list(result))  # [2, 1, 4, 3, 5]