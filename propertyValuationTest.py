from propertyValuation import model, test_loader, criterion
import torch

model.load_state_dict(torch.load('property_valuation_model.pth'))
model.eval()
running_test_loss = 0
total_absolute_error = 0
count=0

with torch.no_grad():
    for features, targets in test_loader:
        predictions = model(features)
        loss = criterion(predictions, targets)
        running_test_loss += loss.item() * features.size(0)

        pred_prices = predictions*100000
        true_prices = targets*100000
        absolute_errors = torch.abs(pred_prices - true_prices)
        total_absolute_error += torch.sum(absolute_errors).item()
        count +=features.size(0)

test_loss = running_test_loss / len(test_loader.dataset)
menan_absolute_error = total_absolute_error / count

print(f'Wynik wczytanego modelu na zbiorze testowym (srednia roznica miedzy predykcja a rzeczywista cena): {menan_absolute_error:,.0f} $')