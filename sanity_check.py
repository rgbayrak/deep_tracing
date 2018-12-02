import matplotlib.pyplot as plt

a = torch.Tensor.numpy(target)
b = torch.Tensor.numpy(data)

b = np.squeeze(b)
a = np.squeeze(a)
plt.imshow(b[53,:,:])
plt.imshow(a[1, 53,:,:])
