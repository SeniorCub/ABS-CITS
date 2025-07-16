student = {'name':'Dorcas', 'age':19, 'courses': ['Math','Computer Science']}

print(student)
print(student['name'])
print(student.get('phone'))

student.update({'name':'Hameedah', 'age':12, 'phone': '555-555-555'})
print(student)

del student['age']
print(student)

print(len(student))
print(student.keys())
print(student.values())
print(student.items())

for key, value in student.items():
    print(key, value)