# Variables
print("This line will be printed.")

x = 1
if x == 1:
    # indented four spaces
    print("x is 1.")

print("Goodbye, World!")

# Data Types
# Numbers
myint = 7
print(myint)
myfloat = 7.0
print(myfloat)
myfloa = float(7)
print(myfloa)
# Strings
mystring = 'hello'
print(mystring)
mysentence = "Don't worry about apostrophes"
print(mysentence)

#
one = 1
two = 2
three = one + two
print(three)

hello = "hello"
world = "world"
helloworld = hello + " " + world
print(helloworld)

# Lists
mylist = []
mylist.append(1)
mylist.append(2)
mylist.append(3)
print(mylist[0]) # prints 1
print(mylist[1]) # prints 2
print(mylist[2]) # prints 3

# prints out 1,2,3
for x in mylist:
    print(x)