'''
You are given a list of integers where every element appears exactly twice, 
except for one element which appears only once. The list can be as large as 10^6 elements. The challenge is to find that one unique element efficiently.

Constraints:

Time Complexity: O(n)
Space Complexity: O(1) (no extra memory usage allowed)
'''
def finds(a) : 
  unique = 0 # only 1 ( no extra memory ) 
  for values  in a: 
    unique ^= values 
  return unique 
a = [1,4,5,7,5,4,7,8,8 ] 
