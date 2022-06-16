import socket
import os
import hashlib
import keyboard
from ImageEncoder import ImageEncoder

if __name__ == "__main__":
    server = socket.socket()
    server.bind(("", 6969))
    server.listen(5)  # listening

    print("Listening starts..")

    while True:
        conn, addr = server.accept()  # wait for connection
        print("Connection:", conn, "\nAddress:", addr)

        while True:
            data = conn.recv(1024)
            if not data:
                print("Client disconnected")
                break

            print("Received commend：", data.decode("utf-8"))
            cmd, filename, q, T = data.decode("utf-8").split(" ")
            q = int(q)
            T = int(T)
            filepath = ".\\out\\image.bit"
            ori_path = ".\\pic\\" + filename + ".512"

            if cmd == "get":
                # Compress image
                print("--------------------------------------")
                print(">START ENCODING...")
                encoder = ImageEncoder(ori_path, q, T)
                encoder.run()
                print("<ENCODING OVER!")
                print("--------------------------------------")

                if os.path.isfile(filepath):
                    # Send the size fo file
                    size = os.stat(filepath).st_size
                    conn.send(str(size).encode("utf-8"))
                    print("Size of file waiting to be sent：%d Bytes" % (size))

                    # Content of the file
                    conn.recv(1024)

                    m = hashlib.md5()
                    f = open(filepath, "rb")
                    for line in f:
                        conn.send(line)
                        m.update(line)
                    f.close()

                    # MD5 checksum
                    md5 = m.hexdigest()
                    conn.send(md5.encode("utf-8"))  # send MD5 value
                    print("--------------------------------------")
                    print("md5:", md5)

    server.close()
