
function[Fitness,Acc,Prec,Sens,Spec,Fs,sFeat,Sf,Nf,curve,best_score,dt]=MLBPOA(feat,label,N,max_Iter,HO,Alpha,Beta)
tic
thres = 0.5; 
fun = @jFitnessFunction;
fun1 = @jFit1;
dim = size(feat,2);
Vmax = 6;
X   = zeros(N,dim);
V   = zeros(N,dim);
for i = 1:N
    for d = 1:dim
        X(i,d) = rand();
    end
end

fit  = zeros(1,N);
fitG = inf;

for i = 1:N
    for d = 1:dim
        if rand() > 0.5
           X(i,d) = 1;
        end
    end
end

for i = 1:N
    fit(i) = fun(feat,label,(X(i,:)),HO);
    if fit(i) < fitG
        Xgb  = X(i,:);
        fitG = fit(i);
    end
end

curve = inf;
t = 1;

for t=1:max_Iter
    [best , location]=min(fit);
    if t==1
        Xbest=X(location,:);                                          
        fbest=best;                                          
    elseif best<fbest
        fbest=best;
        Xbest=X(location,:);
    end
    X_FOOD=[];
    k=randperm(N,1);
    X_FOOD=X(k,:);
    F_FOOD=fit(k);  
    for i=1:N      
        I=round(1+rand(1,1));
        if fit(i)> F_FOOD
            X_new=X(i,:)+ rand(1,1).*(X_FOOD-I.* X(i,:)).*exp((t/max_Iter)*(1-sqrt(t/max_Iter))); %Eq(4)
        else
            X_new=X(i,:)+ rand(1,1).*(X(i,:)-1.*X_FOOD).*exp((t/max_Iter)*(1-sqrt(t/max_Iter))); %Eq(4)
        end
        f_new = fun(feat,label,(X_new),HO);
        if f_new <= fit (i)
            X(i,:) = X_new;
            fit (i)=f_new;
        end
        X_new1=X(i,:)+0.2*(1-t/max_Iter).*(2*rand(1,dim)-1).*X(i,:); 
        beta=2*(exp(1)-exp(t/max_Iter))*sin(2*pi*rand);
                Hprey=X_FOOD+beta*abs(mean(X)-X_FOOD);
                r4=rand;
                Eta=exp(r4*(1-t)/max_Iter)*(cos(2*pi*r4)); 
                X_new2=Hprey+Eta*(Hprey-round(rand)*X(i,:));
       if X_new1 < X_new2
            X_new = X_new1;
       else
            X_new = X_new2;
        end

        for j=1:dim
            XX = X_new(j);
            XX(XX>Vmax)=Vmax;XX(XX<-Vmax);
            X_new(j) = XX;
        end
        V(i,d) = X_new(j);
        
        TF     = 1 / (1 + exp(-1/3.*V(i,d)));
        if TF > rand()
            X1(i,d) = 1;
        else
            X1(i,d) = 0;
        end
        TF     = 1 / (1 + exp(-1/2.*V(i,d)));
        if TF > rand()
            X2(i,d) = 1;
        else
            X2(i,d) = 0;
        end
        TF     = 1 / (1 + exp(-V(i,d)));
        if TF > rand()
            X3(i,d) = 1;
        else
            X3(i,d) = 0;
        end
        TF     = 1 / (1 + exp(-2.*V(i,d)));
        if TF > rand()
            X4(i,d) = 1;
        else
            X4(i,d) = 0;
        end
        
    if rand() >= 0.6
        if t <= 5/10*(max_Iter)
        X11(i,d)=(and( X1(i,d), X2(i,d)));
        X12(i,d)=(and( X11(i,d), X3(i,d)));
        X(i,d)=(and( X12(i,d), X4(i,d)));
        if rand() <= 0.01
            X(i,d)=1- X(i,d);
        end
        else
        X11(i,d)=(or( X1(i,d), X2(i,d)));
        X12(i,d)=(or( X11(i,d), X3(i,d)));
        X(i,d)=(or( X12(i,d), X4(i,d)));
        if rand() <= 0.01
            X(i,d)=1- X(i,d);
        end            
        end
     else
        X11(i,d)=(xor( X1(i,d), X2(i,d)));
        X12(i,d)=(xor( X11(i,d), X3(i,d)));
        X(i,d)=(xor( X12(i,d), X4(i,d)));
        if rand() <= 0.01
            X(i,d)=1- X(i,d);
        end             
     end 
        
        f_new = fun(feat,label,(X_new),HO);
        if f_new <= fit (i)
            X(i,:) = X_new;
            fit (i)=f_new;
        end
    end    
    curve(t)=fbest;
    average(t) = mean (fit);    
end
[CM1,CM2] = fun1(feat,label,(Xbest),HO);
[Acc, Prec, Sens, Fs, Spec] = metrics(CM1,CM2);
best_score=fbest;
Best_pos=Xbest;
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
Nf    = length(Sf);
Fitness=Alpha*best_score+Beta*(Nf/dim);
dt = toc;
end

