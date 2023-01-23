import heapq
import math

# -----
class Heap:
    def __call__(self, elements):
        # 최소 힙
        self.heap = list()
        self.heap.append(None)
        elements.sort()
        self.heap.extend(elements.copy())

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

            if parent < kid:
                self.heap[parent_index] = kid
                self.heap[kid_index] = parent

                check(parent_index)
            else:
                return
        
        check(element_index)


    def heappop(self):
        # 루트 노드가 우선순위가 높으므로 루트부터 삭제
        # 어렵다 내일 하자
        pass


# -----
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

    heap.heappush(5)

    # result = 0
    # while len(heap.heap) >= 2:
    #     min_ = heap.heappop()
    #     if min_ >= _threshold:
    #         return result

    #     else:
    #         min_2 = heap.heappop()
    #         new_min = min_ + min_2
    #         heap.heappush(new_min)
    #         result += 1

    # if heap.heap[0] >= _threshold:
    #     return result
    # else:
    #     return -1


if __name__ == '__main__':

    T = 4
    L = [4, 6, 7, 1, 5, 2, 3]

    # result = check_beaker_liquid(L, T)
    # print(result)

    result_custom = custom_check_beaker_liquid(L, T)
    print(result_custom)

