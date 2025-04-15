import os 
import pickle as pkl

def main(): 
    path = '/home/ubuntu/ehrensberger/master-thesis/master-thesis/metadata.pkl' 

    with open(path, 'rwb') as file: 
        data = pkl.load(file)
        print(type(data))
        print(list(data.keys()))
        data['SAWYER']['MIN_STATES']['X'] = 5
        print(data)


if __name__ == '__main__':
    main()
