import torch


def predict(x, guide, num_samples):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return torch.argmax(mean, dim=1)


def calculate_test_acc(dataloader, predictive):
    correct = 0
    total = 0
    for j, data in dataloader:
        images, labels = data
        samples = {
            k: v.reshape(predictive.num_samples).detach().cpu().numpy()
            for k, v in predictive(images.view(-1, 28 * 28).cuda())
        }
        predicted = predict(images.view(-1, 28 * 28).cuda(), guide, num_samples)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()
    print("accuracy: %d %%" % (100 * correct / total))


def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats
