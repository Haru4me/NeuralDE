import torch
import matplotlib.pyplot as plt

def viz2d(t, true_y, pred_y, func, epoch):

    plt.close()

    t = t.detach().numpy()
    true_y = true_y.squeeze(1).detach().numpy()
    pred_y = pred_y.squeeze(1).detach().numpy()


    n_pts = 50
    x = torch.linspace(-2,2, n_pts)
    y = torch.linspace(-2,2, n_pts)
    X, Y = torch.meshgrid(x, y)
    z = torch.cat([X.reshape(-1,1), Y.reshape(-1,1)], 1)
    f = func.vf(0,z).cpu().detach()
    fx, fy = f[:,0], f[:,1]
    fx, fy = fx.reshape(n_pts , n_pts), fy.reshape(n_pts, n_pts)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.streamplot(X.numpy().T, Y.numpy().T, fx.numpy().T, fy.numpy().T, color='black')
    ax1.contourf(X.T, Y.T, torch.sqrt(fx.T**2+fy.T**2), cmap='RdYlBu')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_title('Vector field')
    ax1.set_xlim(-2,2)
    ax1.set_ylim(-2,2)

    ax2.plot(pred_y[:,0], pred_y[:,1], '--')
    ax2.plot(true_y[:,0], true_y[:,1], '-')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('Phase portrait')

    ax3.plot(t, pred_y, '--')
    ax3.plot(t, true_y, '-')
    ax3.set_ylabel('$x_1$, $x_2$')
    ax3.set_xlabel('$t$')
    ax3.set_title('Trajectories')

    fig.tight_layout()
    plt.savefig(f'./assets/demo_imgs/{epoch}.png')
    plt.draw()
    plt.pause(0.001)
