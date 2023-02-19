import heapq
import math

# -----
class Heap:
    # Custom 최소 Heap 클래스
    def __call__(self, elements):
        self.heap = list()
        self.heap.append(None)
        elements = sorted(elements)
        self.heap.extend(elements)

    # def exchange(self, parent_index, kid_index):
    #     parent = self.heap[parent_index]
    #     kid = self.heap[kid_index]

    #     if parent > kid:
    #         self.heap[parent_index] = kid
    #         self.heap[kid_index] = parent

    def getroot(self):
        return self.heap[1]

    def heappush(self, element):
        # 배열의 마지막에 element를 집어넣고, 부모와 체크해가며 자리를 바꾼다
        # 부모 노드 index를 구하고 싶을 때: (자식 노드 index / 2)
        # 자식 노드 index를 구하고 싶을 때: (부모 노드 index * 2), (부모 노드 index * 2 + 1)
        self.heap.append(element)
        element_index = len(self.heap) - 1

        def check(kid_index):
            if kid_index == 1:
                return

            kid = self.heap[kid_index]
            parent_index = math.floor(kid_index / 2)
            parent = self.heap[parent_index]

            if parent > kid:
                self.heap[parent_index] = kid
                self.heap[kid_index] = parent

                check(parent_index)
            else:
                return
        
        check(element_index)

    def heappop(self):
        # 루트 노드가 우선순위가 높으므로 루트부터 삭제
        # 루트가 삭제된 자리에 완전이진트리의 마지막 노드를 가져오고, 루트 자리에 위치한 새로운 노드를 자식 노드와 비교하며 교환한다
        root = self.heap[1]
        self.heap = [None] + [self.heap[-1]] + self.heap[2:-1]

        def check(parent_index):
            parent = self.heap[parent_index]
            kid_index_1 = parent_index * 2
            kid_index_2 = parent_index * 2 + 1

            # (1) kid가 없는 경우
            if kid_index_1 >= len(self.heap):
                return
            # (2) 왼쪽 kid만 존재하는 경우
            elif kid_index_2 == len(self.heap):
                kid_index = kid_index_1
            # (3) 양쪽 kid가 둘 다 존재하는 경우
            elif kid_index_2 < len(self.heap):
                kid_1 = self.heap[kid_index_1]
                kid_2 = self.heap[kid_index_2]
                kid_index = kid_index_1 if kid_1 <= kid_2 else kid_index_2
            
            kid = self.heap[kid_index]
            if parent > kid:
                self.heap[parent_index] = kid
                self.heap[kid_index] = parent

                check(kid_index)
            else:
                return

        check(1)
        return root

# -----
# heapq 패키지를 이용한 구현
def check_beaker_liquid(_list, _threshold):
    heapq.heapify(_list)

    result = 0
    while len(_list) >= 2:
        min_ = heapq.heappop(_list)
        if min_ >= _threshold:
            return result

        else:
            min_2 = heapq.heappop(_list)
            new_min = min_ + min_2
            heapq.heappush(_list, new_min)
            result += 1
    
    if _list[0] >= _threshold:
        return result
    else:
        return -1


def custom_check_beaker_liquid(_list, _threshold):
    heap = Heap()
    heap(_list)

    result = 0
    while len(heap.heap) >= 2:
        min_ = heap.heappop()
        if min_ >= _threshold:
            return result

        else:
            min_2 = heap.heappop()
            new_min = min_ + min_2
            heap.heappush(new_min)
            result += 1

    if heap.getroot() >= _threshold:
        return result
    else:
        return -1


if __name__ == '__main__':

    T = 4
    L = [4, 6, 7, 1, 5, 2, 3]

    result_custom = custom_check_beaker_liquid(L, T)
    print(result_custom)

