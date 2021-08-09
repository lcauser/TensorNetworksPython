N = 100
time = 0.01
s = 0.0005
maxDim = 100


psiProj = productMPS(2, N, [0.0, 1.0])
psiEdge = copy.deepcopy(psiProj)
psiEdge.tensors[0].storage[0, 0, 0] = 1.0
psiEdge.tensors[0].storage[0, 1, 0] = 0.0
psiEdge2 = copy.deepcopy(psiProj)
psiEdge2.tensors[19].storage[0, 0, 0] = 1.0
psiEdge2.tensors[19].storage[0, 1, 0] = 0.0

psi = productMPS(2, N, [np.sqrt(0.5), np.sqrt(0.5)])
psi0 = copy.deepcopy(psi)
#psi0 = variationalSum(psi0, [psi, (-1.0*dot(psi, psiProj))*psiProj], 4, 1000, 1, 2, 10**-12) 
psi0 = variationalSum(psi0, [psiEdge, psiEdge2], 4, 1000, 1, 2, 10**-20) 
psi0.normalize()

n = np.zeros((2, 2))
n[0, 0] = 1
x = np.zeros((2, 2))
x[0, 1] = 1
x[1, 0] = 1
A = 0.5*(np.exp(-s)*x - np.eye(2))
storage1 = np.zeros((2, 2, 2))
storage1[:, :, 0] = n
storage1[:, :, 1] = A
storage2 = np.zeros((2, 2, 2))
storage2[0, :, :] = A
storage2[1, :, :] = n
ten1 = tensor((2, 2, 2), storage1)
ten2 = tensor((2, 2, 2), storage2)
ten = contract(ten1, ten2, 2, 0)
tenexp1 = exp(ten, [1, 3], time)
tenexp2 = exp(ten, [1, 3], time/2)

loss = 0
oldDim = psi0.maxBondDim()
for j in range(10000):
    for i in range(int(N/2)):
        psi0.orthogonalize(2*i)
        M = contract(psi0.tensors[2*i], psi0.tensors[2*i+1], 2, 0)
        M = trace(contract(M, tenexp2, 1, 0), 1, 4)
        M = permute(M, 1)
        psi0.replacebond(2*i, M)
    
    for i in range(int(N/2)-1):
        psi0.orthogonalize(N-2*i-2)
        M = contract(psi0.tensors[N-2*i-3], psi0.tensors[N-2*i-2], 2, 0)
        M = trace(contract(M, tenexp1, 1, 0), 1, 4)
        M = permute(M, 1)
        psi0.replacebond(N-2*i-3, M, True)
        
    for i in range(int(N/2)):
        psi0.orthogonalize(2*i)
        M = contract(psi0.tensors[2*i], psi0.tensors[2*i+1], 2, 0)
        M = trace(contract(M, tenexp2, 1, 0), 1, 4)
        M = permute(M, 1)
        psi0.replacebond(2*i, M)
        
    psiOld = copy.deepcopy(psi0)
    psi0 = variationalSum(psi0, [psi0, (-1.0*dot(psi0, psiProj))*psiProj], 4, 20, 1, min(2*oldDim, maxDim), 10**-12)
    loss = abs(1 - dot(psi0, psiOld) / dot(psiOld, psiOld))
    normal = psi0.norm()
    psi0.normalize()
    oldDim = psi0.maxBondDim()
    print("Time="+str(round((j+1)*time, 3))+" maxbonddim="+str(oldDim)+
          " norm="+str(normal)+" loss="+str(loss))