#Image recognition granting access code

inp = ['NotKey', 'NotKey', 'NotKey', 'Key', 'NotKey', 'NotKey', 'NotKey', 'NotKey'] #image labels being inputted
key = ['Key'] #image labels that should grant access
accessGranted = [] #empty list to save the images that the computer grants access to

def compareLabels(inp, key):
    for idx, item in enumerate(key):
        for i in range(len(inp)-1):
            print(item)
            print(i)
            print(idx)
            if key[idx] == inp[i]:
                print('Access granted')
                accessGranted.append(item)
                print(accessGranted)
            else:
                print('Access denied')

#function that runs thru items in one list to see if those items appear in another list            

compareLabels(inp, key)