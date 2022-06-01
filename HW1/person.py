# -*- coding: utf-8 -*-

class person:
  def __init__(self, name, age_yrs , height):
    self.name = name
    self.age_yrs  = age_yrs 
    self.height = height
  
  def __repr__(self):
    return "{:} is {:} years old and {:} cm tall.".format(self.name, self.age_yrs , self.height)
