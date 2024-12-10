from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm


class RecursiveModel(object):
    def __init__(self, bptt_y, all_lags):
        self.bptt_y = bptt_y
        self.all_lags = all_lags

    def setting_model(
        self, model, optimizer, criterion, lr_scheduler, sr_scheduler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.sr_scheduler = sr_scheduler
        self.lr_scheduler = lr_scheduler
        self.device = model.device

    def train(
        self,
        model_path,
        trainloader,
        validloader,
        epochs,
        clipping_value=1.0,
        log_interval=1,
    ):
        best_train_loss = float("inf")
        best_val_loss = float("inf")
        best_epoch = 0
        loss_name = self.criterion._get_name()

        losses = {"training": [], "valid": []}

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        print(
            f"Training until validation scores don't improve for {log_interval} rounds"
        )
        for epoch in range(1, epochs + 1):
            # train
            if self.sr_scheduler is None:
                train_loss = self._train_on_batch(trainloader, clipping_value)
            else:
                train_loss = self._train_scheduled_sampling(trainloader, clipping_value)
                current_sr = self.sr_scheduler.sampling_rate
                self.sr_scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_scheduler.step()

            # valid
            _, val_loss = self.evaluate_recurcive(validloader)

            losses["training"].append(train_loss)
            losses["valid"].append(val_loss)

            if val_loss < best_val_loss:
                torch.save(self.model.state_dict(), model_path)
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_epoch = epoch

            if epoch % log_interval == 0:
                print(
                    "[{}] \t training's {}: {:5.5f} \t valid's {}: {:5.5f} \t | lr={:5.5f}".format(
                        epoch, loss_name, train_loss, loss_name, val_loss, current_lr,
                    ),
                    end="",
                )
                if self.sr_scheduler is None:
                    print()
                else:
                    print(" \t sr={:5.5f}".format(current_sr))

        print("\nEarly stopping, best iteration is:")
        print(
            "[{}] \t training's {}: {:5.5f} \t valid's {}: {:5.5f}".format(
                best_epoch, loss_name, best_train_loss, loss_name, best_val_loss
            )
        )

        self.model.load_state_dict(torch.load(model_path))
        return losses

    def _train_on_batch(self, trainloader, clipping_value=1.0):
        self.model.train()
        total_loss = 0.0

        for (
            en_input_batch,
            en_cat_input_batch,
            de_input_batch,
            de_cat_input_batch,
            y_batch,
        ) in trainloader:

            en_input_batch = en_input_batch.transpose(0, 1)
            de_input_batch = de_input_batch.transpose(0, 1)

            en_cat_input_batch = en_cat_input_batch.transpose(0, 1)
            de_cat_input_batch = de_cat_input_batch.transpose(0, 1)

            y_batch = y_batch.transpose(0, 1).to(self.device).float()

            self.optimizer.zero_grad()
            output = self.model(
                en_input_batch, en_cat_input_batch, de_input_batch, de_cat_input_batch
            )
            loss = self.criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_value)
            self.optimizer.step()

            total_loss += y_batch.size(0) * y_batch.size(1) * loss.item()

        total_loss /= len(trainloader) * trainloader.batch_size * y_batch.size(0)
        return total_loss

    def _train_scheduled_sampling(self, trainloader, clipping_value=1.0):
        self.model.train()
        total_loss = 0.0

        for (
            en_input_batch,
            en_cat_input_batch,
            de_input_batch,
            de_cat_input_batch,
            y_batch,
        ) in trainloader:

            en_input_batch = en_input_batch.transpose(0, 1)
            de_input_batch = de_input_batch.transpose(0, 1)

            en_cat_input_batch = en_cat_input_batch.transpose(0, 1)
            de_cat_input_batch = de_cat_input_batch.transpose(0, 1)

            y_batch = y_batch.transpose(0, 1).to(self.device).float()

            self.optimizer.zero_grad()

            output = self._predict_scheduled_sampling(
                en_input_batch, en_cat_input_batch, de_input_batch, de_cat_input_batch,
            )

            loss = self.criterion(output, y_batch)
            # loss = self.criterion(output, y_batch) + self.criterion(
            #     output.std(0), y_batch.std(0)
            # )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipping_value)
            self.optimizer.step()

            total_loss += y_batch.size(0) * y_batch.size(1) * loss.item()

        total_loss /= len(trainloader) * trainloader.batch_size * y_batch.size(0)
        return total_loss

    def evaluate(self, evalloader):
        total_loss = 0.0
        output = list()
        self.model.eval()  # Turn on the evaluation mode

        with torch.no_grad():
            for (
                en_input_batch,
                en_cat_input_batch,
                de_input_batch,
                de_cat_input_batch,
                y_batch,
            ) in evalloader:

                en_input_batch = en_input_batch.transpose(0, 1)
                de_input_batch = de_input_batch.transpose(0, 1)

                en_cat_input_batch = en_cat_input_batch.transpose(0, 1)
                de_cat_input_batch = de_cat_input_batch.transpose(0, 1)

                y_batch = y_batch.transpose(0, 1).to(self.device).float()

                output_batch = self.model(
                    en_input_batch,
                    en_cat_input_batch,
                    de_input_batch,
                    de_cat_input_batch,
                )
                total_loss += (
                    y_batch.size(0)
                    * y_batch.size(1)
                    * self.criterion(output_batch, y_batch).item()
                )
                output.append(output_batch.transpose(1, 0).squeeze(-1))
        total_loss /= len(evalloader) * evalloader.batch_size * y_batch.size(0)

        output = torch.cat(output, 0).to(self.device).detach().cpu().numpy()

        return output, total_loss

    def evaluate_recurcive(self, evalloader):
        total_loss = 0.0
        output = list()
        self.model.eval()

        with torch.no_grad():
            output = list()
            for (
                en_input_batch,
                en_cat_input_batch,
                de_input_batch,
                de_cat_input_batch,
                y_batch,
            ) in evalloader:

                en_input_batch = en_input_batch.transpose(0, 1)
                de_input_batch = de_input_batch.transpose(0, 1)

                en_cat_input_batch = en_cat_input_batch.transpose(0, 1)
                de_cat_input_batch = de_cat_input_batch.transpose(0, 1)

                y_batch = y_batch.transpose(0, 1).to(self.device).float()

                memory = self.model(src=en_input_batch, src_cat_idx=en_cat_input_batch)

                for future_i in range(self.bptt_y):
                    output_batch = self.model(
                        memory=memory,
                        tgt=de_input_batch,
                        tgt_cat_idx=de_cat_input_batch,
                    )

                    if future_i < self.bptt_y - 1:
                        de_input_batch = self._update_de_input(
                            de_input_batch, output_batch, future_i
                        )

                total_loss += (
                    y_batch.size(0)
                    * y_batch.size(1)
                    * self.criterion(output_batch, y_batch).item()
                )
                output.append(output_batch.transpose(1, 0).squeeze(-1))

            total_loss /= len(evalloader) * evalloader.batch_size * y_batch.size(0)
            output = torch.cat(output, 0).detach().cpu().numpy()

        return output, total_loss

    def predict(self, testloader):
        self.model.eval()

        with torch.no_grad():
            output = list()
            for (
                en_input_batch,
                en_cat_input_batch,
                de_input_batch,
                de_cat_input_batch,
            ) in tqdm(testloader, total=len(testloader)):

                en_input_batch = en_input_batch.transpose(0, 1)
                de_input_batch = de_input_batch.transpose(0, 1)

                en_cat_input_batch = en_cat_input_batch.transpose(0, 1)
                de_cat_input_batch = de_cat_input_batch.transpose(0, 1)

                memory = self.model(src=en_input_batch, src_cat_idx=en_cat_input_batch)

                for future_i in range(self.bptt_y):
                    output_batch = self.model(
                        memory=memory,
                        tgt=de_input_batch,
                        tgt_cat_idx=de_cat_input_batch,
                    )

                    if future_i < self.bptt_y - 1:
                        de_input_batch = self._update_de_input(
                            de_input_batch, output_batch, future_i
                        )

                output.append(output_batch.transpose(1, 0).squeeze(-1))

            output = torch.cat(output, 0).detach().cpu().numpy()

        return output

    def _predict_scheduled_sampling(
        self, en_input_batch, en_cat_input_batch, de_input_batch, de_cat_input_batch,
    ):

        self.model.eval()
        with torch.no_grad():
            memory = self.model(src=en_input_batch, src_cat_idx=en_cat_input_batch)
            for future_i in range(self.bptt_y):
                if np.random.binomial(1, self.sr_scheduler.sampling_rate, 1)[0] == 1:
                    continue
                output_batch = self.model(
                    memory=memory, tgt=de_input_batch, tgt_cat_idx=de_cat_input_batch
                )

                if future_i < self.bptt_y - 1:
                    de_input_batch = self._update_de_input(
                        de_input_batch, output_batch, future_i
                    )

        self.model.train()
        output_batch = self.model(
            en_input_batch, en_cat_input_batch, de_input_batch, de_cat_input_batch
        )

        return output_batch

    def _update_de_input(self, de_input, future_y, future_i):
        idx_0 = []
        idx_2 = []
        for lag, dim_i in enumerate(range(future_i, self.bptt_y - 1), start=1):
            if lag not in self.all_lags:
                continue

            dim_j = self.all_lags.index(lag)
            idx_0.append(dim_i + 1)
            idx_2.append(dim_j)

        idx_0 = torch.tensor(idx_0)
        idx_2 = torch.tensor(idx_2)
        de_input[idx_0, :, idx_2] = future_y[future_i, :, 0].cpu()

        return de_input

    def load_state_dict(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
