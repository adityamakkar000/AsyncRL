from multiprocessing.managers import BaseManager
import time

class QueueManager(BaseManager):
    pass

# Register the same name so the manager knows what to look for
QueueManager.register('get_task_queue')

def connect_to_queue():
    # Replace '127.0.0.1' with the Server's IP address if on a different machine
    manager = QueueManager(address=('127.0.0.1', 50000), authkey=b'abracadabra')
    manager.connect()
    
    # Get the proxy object
    q = manager.get_task_queue()
    
    # Send a Python object (a dictionary in this case)
    # data = {'id': 3, 'message': 'Hello from the client!'}
    # print(f"Sending: {data}")
    # q.put(data)

    print(q.qsize())  # Check the size of the queue

    while q.qsie() < train sie:
        ... 
        sync_host() 

    batch_size_per_host = batch_size // n_hosts 
    for i in range(n_hosts):
        on_this_device_data = []
        for k in in range(batch_size_per_host):
            if jax.processs_index() == i:
                on_this_device_data.append(q.get())
    

        for i in range(n_hosts):

    # p = q.get()
    # print(f"Received: {p}")

if __name__ == "__main__":
    connect_to_queue()