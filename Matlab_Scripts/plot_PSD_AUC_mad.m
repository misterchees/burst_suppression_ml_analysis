function [AUC,ax] = plot_PSD_AUC_mad (Freq, PSD1,PSD2,c1,c2,cs1,cs2)
%c1,c2 are the colors

% c1='b';
% cs1=[192 216 227]/255;
% c2='r';
% cs2=[255 204 203]/255;
BUF1=PSD1; clear PSD1
BUF2=PSD2; clear PSD2

for a=1:size(BUF1,1)
    % Debug-Check for inputs
    checkMesInputs(a, BUF1, BUF2);

    AUC_BUF=mes(BUF1(a,:)',BUF2(a,:)','auroc','isDep',0,'nBoot',10e3);
    AUC(a,1)=AUC_BUF.auroc; AUC(a,2)=AUC_BUF.aurocCi(1); AUC(a,3)=AUC_BUF.aurocCi(2); clear AUC_BUF
    AUC(a,4)=ranksum(BUF1(a,:)',BUF2(a,:)');
end

figure
ax(1)=subplot(3,1,[1:2])
hold off
B1_m=nanmedian(BUF1,2); B1_mad=mad(BUF1',1)';
B2_m=nanmedian(BUF2,2); B2_mad=mad(BUF2',1)';

% plot(Freq, 10*log10(BUF1),'o','markeredgecolor', c1)
% % plot(Freq, 10*log10(BUF1),'o','markeredgecolor', c1,'markerfacecolor',c1)
% hold on
% plot(Freq, 10*log10(BUF2),'o','markeredgecolor', c2)
% % plot(Freq, 10*log10(BUF2),'o','markeredgecolor', c2,'markerfacecolor',c2)
% plot(Freq,10*log10(B1_m),'-','color',c1,'linewidth',4)
% plot(Freq,10*log10(B2_m),'-','color',c2,'linewidth',4)

X=[Freq;flipud(Freq)];Y=[10*log10((B1_m+B1_mad));flipud(10*log10(B1_m-B1_mad))];
fill(X',Y'',cs1,'LineStyle','none');clear BUF X Y

hold on

X=[Freq;flipud(Freq)];Y=[10*log10((B2_m+B2_mad));flipud(10*log10(B2_m-B2_mad))];
fill(X',Y'',cs2,'LineStyle','none');clear BUF
plot(Freq,10*log10(B1_m),'-','color',c1,'linewidth',2)
plot(Freq,10*log10(B2_m),'-','color',c2,'linewidth',2)


box off
xlim([0.5 47])
%ylim([-45 -5])
ylabel('Power [dB]')
xlabel('Frequency [Hz]')
set(gca,'FontName','Arial','FontSize',12,'FontWeight','Bold', 'LineWidth', 2);
%   title(tit)

ax(2) = subplot(313);
hold off


plot(Freq,AUC(:,1),'ko')
hold on
line([1 50],[.5 .5])

plot(Freq,AUC(:,2),'kx')
plot(Freq,AUC(:,3),'kx')

% AUC > 0.75 in gray
BUF_A=AUC(:,1); BUF_A(BUF_A<0.7)=nan;
BUF_B=AUC(:,1); BUF_B(BUF_B>0.3)=nan;
plot(Freq,BUF_A,'o','markeredgecolor', [.7 .7 .7],'markerfacecolor',[.7 .7 .7]); clear BUF_A
plot(Freq,BUF_B,'o','markeredgecolor', [.7 .7 .7],'markerfacecolor',[.7 .7 .7]); clear BUF_B

% line([5 5],[0 1])
% line([9 9],[0 1])
% line([15 15],[0 1])
% line([23 23],[0 1])

% AUC sign in black
BUF_A=AUC(:,2); BUF_A(BUF_A>0.5)=1;BUF_A(BUF_A<0.5)=-1; BUF_A(BUF_A==0.5)=0;
BUF_B=AUC(:,3); BUF_B(BUF_B>0.5)=1;BUF_B(BUF_B<0.5)=-1; BUF_B(BUF_B==0.5)=0;
BUF_AB=BUF_A.*BUF_B; clear BUF_A BUF_B
BUF_C=AUC(:,1); BUF_C(BUF_AB~=1)=nan;
plot(Freq,BUF_C,'ko','markerfacecolor','k'); clear BUF_C
box off
xlim([0.5 47])
ylim([0 1])
ylabel('AUC')
xlabel('Frequency [Hz]')
set(gca,'FontName','Arial','FontSize',12,'FontWeight','Bold', 'LineWidth', 2);
set(gcf, 'Position', [10, 10, 550, 900]);
end

function checkMesInputs(rowIdx, BUF1, BUF2)
    % Diagnosefunktion für Inputs von mes()
    %
    % :param rowIdx: aktuelle Zeile (Index a)
    % :param BUF1: Matrix 1 (Samples x Frequenzen)
    % :param BUF2: Matrix 2 (Samples x Frequenzen)

    fprintf('\n--- Diagnose für Zeile %d ---\n', rowIdx);
    fprintf('BUF1 size = [%s], BUF2 size = [%s]\n', ...
        mat2str(size(BUF1)), mat2str(size(BUF2)));

    if rowIdx > size(BUF1,1) || rowIdx > size(BUF2,1)
        error('Index %d überschreitet die Dimensionen von BUF1 oder BUF2!', rowIdx);
    end

    slice1 = BUF1(rowIdx,:)';
    slice2 = BUF2(rowIdx,:)';

    fprintf('BUF1 slice (erste 10 Werte):\n');
    disp(slice1(1:min(10,end)));
    fprintf('BUF2 slice (erste 10 Werte):\n');
    disp(slice2(1:min(10,end)));

    if isempty(slice1) || isempty(slice2)
        error('Leere Eingabe in Zeile %d! BUF1 oder BUF2 Slice ist leer.', rowIdx);
    end
    if all(isnan(slice1)) || all(isnan(slice2))
        error('Nur NaNs in Zeile %d! BUF1 oder BUF2 Slice unbrauchbar.', rowIdx);
    end
end