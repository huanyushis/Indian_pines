import datetime

year = int(input("请输入4位数的年份:"))
month = int(input("请输入月份:"))
day = int(input("请输入当月哪一天:"))

targetDay = datetime.date(year, month, day)
print("%s是%s年的第%s天." % (targetDay, targetDay.year))
