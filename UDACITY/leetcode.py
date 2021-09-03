def removeDuplicates(nums):
    i = 0 
    while i < len(nums) - 1:
        if nums[i] == nums[i+1]:
            del nums[i]
        else:
            i += 1
    return nums


lst = [1,1,4,5,3,5,6]
print(removeDuplicates(lst))






def removeDuplicates2(nums):
    i = 0
    while i < len(nums) - 1:
        if nums[i] == nums[i+1]:
            del nums[i]
        else:
            i += 1
    return "The length of arrray is {} and elements are {}".format(len(nums), nums)

lst2 = [1,1,4,5,3,5,5,5,5,5,5]
print(removeDuplicates2(lst2))





def maxProfit(prices):
    sum = 0
    i = 0
    for i in range(1, len(prices)):
        #print(len(prices))
        sum += max(prices[i] - prices[i-1],0)
    print(sum)

maxProfit(lst)
