def hello_func():
    print('Hello Function!')

hello_func()

def string_func():
    return 'string function'
print(string_func().upper())

def arg_func(greeting, name='You'):
    return '{}, {}'.format(greeting, name)
print(arg_func('Hi', name='Dorcas'))

def sutudent_info(*args, **kwargs):
    print(args)
    print(kwargs)

courses = ['Maths','Physics', 'Chemistry', 'Biology']
info = {'name':'Dorcas', 'age':19, 'gender':'female'}
sutudent_info(*courses, **info)