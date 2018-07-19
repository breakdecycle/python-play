import random



def main():
    char = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()"

    len = int(input('Password Length?:'))
    num = int(input('How many to generate? (suggest more than 3):'))


    for i in range(num):
      pass_gen = ''

      for j in range(len):
        pass_gen += random.choice(char)
      
      print (pass_gen)
  
  
if  __name__ == '__main__':
    main()
