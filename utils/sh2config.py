"""convert the command-line-arguments to a vs-code-debug-configuration strin"""

s = input('Enter sh-string: ')
arr = s.split()

# change exp-name to test
idx_name = arr.index('--exp_name')
arr[idx_name+1] = 'test'

# remove call to main-file
if arr[0] == 'python' or arr[0] == 'python3': 
    arr = arr[2:]

# add quotation-marks and newlines
s_new = ' '.join(arr)
print("\n\n\"" + s_new.replace(" ", "\", \"").replace(" \"--", "\n\"--") + "\"\n\n\n")