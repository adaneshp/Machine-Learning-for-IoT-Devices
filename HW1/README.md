# Homework 1

- This code was developed in February, 2022.

In this assignment you will gain some familiarity with the git revision control system and the Python programming language.  These are both core infrastructure that will be necessary to succeed in the rest of the class.
## Programming Tasks

 - The repo includes a starter script named hw1_template.py.  Copy it into a file hw1_complete.py and run it. It should print out several names and tell how long each name is.  It will also create a 
 - Use a list comprehension to create a list of lengths of the names.  Here is an example list comprehension.  This is quite a powerful feature; search around the web for more complex examples.
 ```
new_list = [f(x) for x in original_list]
```
- Create a separate file, named "person.py". In that file, define a class "person" with fields "name", "age", and "height."  Define the __init__ method (constructor) that takes arguments 'name', 'age', and 'height', and sets the object variables of the same names.  Eg. 
```
new_person = person(name='Joe', age=34, height=184)
print("{:} is {:} years old.".format(new_person.name, new_person.age)))
```
should create an object named "new_person" with the name, age, and height indicated and then print out "Joe is 34 years old."

- Add a  method "__repr__(self)" that returns a string of this form: "Bob is 17 years old and 182 cm tall".
- Back in the main file hw1p1.py, add a line to "import person".  Then iterate through the names and create a dictionary named "people" where the key is a name, and the value is a person object.
- Use numpy.array() to convert the list of ages into an array.  Repeat for the list of heights.
Use the numpy.mean() function to find the average age of the people in your list and put that average age in a variable named average_age.
Use matplotlib to create a scatter plot of ages (on the x axis) and heights (on the y axis).  Add gridlines, x and y axis labels, and a title.  Export the plot as a png file. 
- There is already code in the template to load the 'iris' dataset and to make one plot of it.  Explore the data.
- Write a function named classify_iris() that implements a linear classifier for the iris dataset. You'll have a 3x4 array of weights and a 3-element vector of biases. The function will take a 4-D vector of features as its only argument and return an integer label (0,1, or 2).
    - The classifier should effectively implement the function
    
        $y=\operatorname{argmax}(W x+b)$
    - Use numpy's argmax() function and either np.matmul() or the '@' operator to peform matrix-vector multiplication.
- Call the evaluate_classifier() function on your classifier to measure its performance.  The function in already inlcuded in the template file.  In the template it is called on a baseline random-choice classifier, which should on average achieve 33% accuracy.  By default evaluate_classifier also prints out a confusion matrix, which may help you adjust your weights and biases.
- Aim for at least 65% accuracy.  You can either adjust your weights manually, or you can look into functions in scikit-learn or other toolkits that can fit a linear classifier.
- Test your code with pytest.  From your working directory just run 'pytest'.  Pytest will report how many tests are passed. If you have done all of the tasks correctly, you should see six tests passed and none failed.  The tests are defined in 'hw1_test.py'.  Feel free to read the code to see what is tested, but do not modify hw1_test.py.  An identical test file will be automatically run on your code when you check it in, which will determine a component of your homework grade.
