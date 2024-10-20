student_scores = {"Alice": 88, "Bob":75, "Amaan":88, "Rohit": 93, "Hardik":93,  "Deeksha":93}
highest_score = max(student_scores.values())
toppers = [name for name, score in student_scores.items()
if score ==  highest_score]
print("By Amaan Shaikh S.")
print('Students with highest score:')
print(f'Highest Score : {highest_score}')
print(f'Student Names : {toppers}')