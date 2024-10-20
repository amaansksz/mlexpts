def find_Common_elements(list1, list2):
   # Convert lists into set to remove duplicates
   set1 = set(list1)
   set2 = set(list2)
   #Find intersection of two sets to get common elements
   common_elements_set = set1.intersection(set2)
   #Convert set back to sorted list
   common_elemets_list = sorted(list(common_elements_set))
   return common_elemets_list
#Given lists:
list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
list2 = [5, 6, 7, 8, 9, 10, 11, 12, 13]
#Find and print common elements
Common_elements = find_Common_elements(list1, list2)
print("Name : Amaan Shaikh S.")
print("UIN : 211P052 ")
print("Common elements are: ", Common_elements)