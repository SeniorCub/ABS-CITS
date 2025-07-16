month_days=[0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def is_leap(year):
    # return True for leap year, False for non-leap years
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def day_in_month(year, month):
    # Return number of days in that month in that year
    if not 1 <= month <= 12:
        return 'Invalid Month'

    if month == 2 and is_leap(year):
        return 29

    return month_days[month]

print(is_leap(2006))
print(day_in_month(2017, 2))