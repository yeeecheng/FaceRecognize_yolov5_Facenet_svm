import socket 
import os 
import argparse




def sent_dataset(conn,data):
    
    path =data
    print(path)
    for r ,dirs ,files in os.walk(path):
        for f in files:
            
            file_path = os.path.join(r,f)
            #print(file_path)
            send_file(conn,file_path,"dataset")

      
def send_file(conn,path,mode):
   
    conn.send(mode.encode())
    reply = conn.recv(1024)
    if reply.decode() == "get mode":
        file = open(path,"rb")
        file_bytes = file.read()
        path_fragment = path.split("/")
        
        if mode == "dataset":
            send_path = "dataset/"+path_fragment[-3]+"/"+path_fragment[-2]+"/"+path_fragment[-1]
        elif mode == "yaml":
            send_path =path_fragment[-1]
        
        conn.send("{}|{}".format(len(file_bytes),"./"+send_path).encode())
        reply = conn.recv(1024)
        if reply.decode() == "ok":
            total_size = len(file_bytes)
            cur =0
            while cur < total_size:
                data = file_bytes[cur:cur+1024]
                conn.send(data)
                cur += len(data)
            reply = conn.recv(1024)
           
            if reply.decode() == "success":
                
                print("success")
                
            file.close()
            
def receive_data(conn,root):
    
    while True:
        
        mode =conn.recv(1024).decode()
        
        if mode == "end":
            break
        conn.send(b'get mode')
        
        if mode  == "tflite":
            
            info = conn.recv(1024)
            size ,file_path = info.decode().split('|')
            file_path = os.path.join(root,"./"+file_path.split("/")[6]) 
            
            if size and file_path:
                
                new_file = open(file_path,"wb")
                conn.send(b'ok') 
                
                file =b''
                total_size  = int(size)
                
                get_size = 0
                while(get_size < total_size):
                    
                    data = conn.recv(1024)
                    file+=data
                    get_size +=len(data)
                    
                new_file.write(file[:])
                new_file.close()
                conn.send(b'success')
        
        else :
            msg = conn.recv(1024).decode()
            print(msg)
            
        
def create_client(opt):
    
    TARGET_IP = opt.ip
    PORT = opt.port
    root = opt.root
    
    # 建立連線
    print(f"start connect to {TARGET_IP} {PORT}")
    c = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    c.connect((TARGET_IP,PORT))
    
    # 傳送 dataset
    sent_dataset(c,opt.data)
    # 傳送 yaml
    send_file(c,os.path.join(root,"./dataset.yaml"),"yaml")
    c.send(b'end')
    print("send data finish")
    

    
    # 接受edgetpu_tflite
    receive_data(c,root)
    
    c.close()
                           


    
def main(opt):
    create_client(opt)
    
def parse_opt(known=False):
   
    ROOT =os.getcwd()
    parser =argparse.ArgumentParser()
    parser.add_argument("--root",type=str , default=ROOT )
    parser.add_argument("--data",type=str ,required=True)
    parser.add_argument("--ip",type=str ,required=True) 
    parser.add_argument("--port",type=int,required=True)
   
    return parser.parse_args()[0] if known else parser.parse_args()


def run(**kwargs):

    opt = parse_opt(True)
    for k,v in kwargs.items():
        setattr(opt,k,v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
