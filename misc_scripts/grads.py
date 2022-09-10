        try:
            grads = []
            for param in self.encoder.parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            print("grads", grads.isnan().sum())
        except:
            print("no grads")

