# Indice formule estratte dalla presentazione

PNG generati da `presentazione/presentazione.tex` (font identici al deck:
Times/newtx), 300 dpi, sfondo bianco, testo nero. Nome file =
`slideNN_formulaM.png` con NN numero di slide.

- `slide02_formula1.png` — slide 2, formula 1: `J(w) = \frac{1}{N}\sum_{i=1}^{N}\ell(w;i), \qquad w\in\mathbb{R}^m`
- `slide04_formula1.png` — slide 4, formula 1: `g_k = \nabla\ell(w_k;i_k) - c_k + \mathbb{E}[c_k],`
- `slide05_formula1.png` — slide 5, formula 1: `g_k=\nabla\ell(w_k;i_k)-\nabla\ell(\bar w;i_k)+\nabla J(\bar w)`
- `slide06_formula1.png` — slide 6, formula 1: `g_k=\nabla\ell(w_k;i_k)-g^{(i_k)}+\bar g`
- `slide08_formula1.png` — slide 8, formula 1: `\mathcal{V}:=\frac{1}{N}\sum_{i=1}^{N}\bigl(\nabla\ell(w_k;i)-\nabla J(w_k)\bigr)^2,\qquad \widehat{\mathcal{V}}:=\frac{1}{n_k-1}\sum_{i\in\mathcal{S}_k} \bigl(\nabla\ell(w_k;i)-g_k\bigr)^2 ,`
- `slide09_formula1.png` — slide 9, formula 1: `\mathrm{Var}(g_k) =\mathrm{Var}\!\Bigl(\frac{1}{n_k}\sum_{i\in\mathcal{S}_k} \nabla\ell(w_k;i)\Bigr) =\frac{1}{n_k^2}\sum_{i\in\mathcal{S}_k}\mathrm{Var}\bigl(\nabla \ell(w_k;i)\bigr) =\frac{\mathcal{V}}{n_k}.`
- `slide09_formula2.png` — slide 9, formula 2: `\mathrm{Var}(g_k)=\frac{\mathcal{V}}{n_k}\cdot\frac{N-n_k}{N-1}.`
- `slide10_formula1.png` — slide 10, formula 1: `\frac{\|\widehat{\mathcal{V}}\|_1}{n_k} \;\le\; \theta^2\,\|g_k\|_2^2, \qquad \theta\in(0,1)`
- `slide10_formula2.png` — slide 10, formula 2: `n_k^{\mathrm{new}} \;=\; \left\lceil \frac{\|\widehat{\mathcal{V}}\|_1}{\theta^2\,\|g_k\|_2^2} \right\rceil`
- `slide11_formula1.png` — slide 11, formula 1: `\nabla J(w_k)^\top g_k =\|g_k\|_2^2-e_k^\top g_k \;\ge\; \|g_k\|_2^2-\|e_k\|_2\|g_k\|_2 \;\ge\; (1-\theta)\|g_k\|_2^2 > 0 .`
- `slide11_formula2.png` — slide 11, formula 2: `\mathbb{E}\bigl[\nabla J(w_k)^\top g_k\bigr] =\nabla J(w_k)^\top\mathbb{E}[g_k] =\|\nabla J(w_k)\|_2^2 > 0,`
- `slide13_formula1.png` — slide 13, formula 1: `\boxed{\,f(x+h)-f(x)\le c_1\,\nabla f(x)^Th\,},\qquad 0<c_1<1.`
- `slide13_formula2.png` — slide 13, formula 2: `\boxed{\,\nabla f(x+h)^Th\ge c_2\,\nabla f(x)^Th\,},\qquad c_1<c_2<1.`
- `slide13_formula3.png` — slide 13, formula 3: `\boxed{\;\begin{cases} f(x+h)-f(x)\le c_1\nabla f(x)^Th & \text{(caduta sufficiente)}\\[2pt] \nabla f(x+h)^Th\ge c_2\nabla f(x)^Th & \text{(curvatura)} \end{cases}\;}`
- `slide15_formula1.png` — slide 15, formula 1: `\nabla^2 J_{\mathcal{H}_k}(w_k)\,d=-\nabla J_{\mathcal{S}_k}(w_k).`
- `slide15_formula2.png` — slide 15, formula 2: `R=\frac{|\mathcal{H}_k|}{|\mathcal{S}_k|}<1, \qquad |\mathcal{H}_k|=\bigl\lceil R\,|\mathcal{S}_k|\bigr\rceil .`
- `slide16_formula1.png` — slide 16, formula 1: `r_k=\nabla^2J_{\mathcal{H}_k}(w_k)\,d+\nabla J_{\mathcal{S}_k}(w_k),`
- `slide16_formula2.png` — slide 16, formula 2: `\nabla^2J_{\mathcal{S}_k}(w_k)d+\nabla J_{\mathcal{S}_k}(w_k) =\underbrace{r_k}_{\text{residuo del CG}} +\underbrace{\Delta_{\mathcal{H}_k}(w_k;d)}_{\text{errore di Hessiana}},`
- `slide17_formula1.png` — slide 17, formula 1: `\mathbb{E}\bigl[\|\Delta_{\mathcal{H}_k}(w_k;d)\|_2^2\bigr] \approx\frac{\bigl\|\mathrm{Var}_{i\in\mathcal{H}_k} \bigl(\nabla^2\ell(w_k;i)\,d\bigr)\bigr\|_1}{|\mathcal{H}_k|}.`
- `slide18_formula1.png` — slide 18, formula 1: `\mathrm{Var}_{i\in\mathcal{H}_k}\bigl(\nabla^2\ell(w_k;i)(\lambda d)\bigr) =\lambda^2\,\mathrm{Var}_{i\in\mathcal{H}_k}\bigl(\nabla^2\ell(w_k;i)d\bigr).`
- `slide18_formula2.png` — slide 18, formula 2: `\alpha:=\frac{\bigl\|\mathrm{Var}_{i\in\mathcal{H}_k}\bigl(\nabla^2 \ell(w_k;i)\,p_0\bigr)\bigr\|_1}{\|p_0\|_2^2} =\frac{\bigl\|p_0^T\Sigma_{\mathcal{H}_k}p_0\bigr\|_1}{\|p_0\|_2^2},`
- `slide19_formula1.png` — slide 19, formula 1: `\|r_{j+1}\|_2^2\;\le\;\Psi(d_j)=\gamma\,\|d_j\|_2^2.`
- `slide21_formula1.png` — slide 21, formula 1: `[\widetilde{\nabla}F(w)]_i =\underbrace{\frac{\partial J(w)}{\partial w_i}}_{\text{gradiente di }J} +\underbrace{g_i}_{\text{subgradiente di }\nu|w_i|},\qquad g_i\in\begin{cases} \{+\nu\}, & w_i>0,\\ \{-\nu\}, & w_i<0,\\ [-\nu,+\nu], & w_i=0. \end{cases}`
- `slide22_formula1.png` — slide 22, formula 1: `z_k^i=\begin{cases} +1 & w_k^i>0 \ \text{oppure}\ \bigl(w_k^i=0 \text{ e } \partial J_{\mathcal{S}_k}/\partial w_i<-\nu\bigr)\\[2pt] -1 & w_k^i<0 \ \text{oppure}\ \bigl(w_k^i=0 \text{ e } \partial J_{\mathcal{S}_k}/\partial w_i>\nu\bigr)\\[2pt] 0 & \text{altrimenti} \end{cases}`
- `slide23_formula1.png` — slide 23, formula 1: `P(w_i)=\begin{cases} w_i & \operatorname{sign}(w_i)=\operatorname{sign}(z_k^i)\\ 0 & \text{altrimenti} \end{cases}`
- `slide23_formula2.png` — slide 23, formula 2: `F_{\mathcal{S}_k}\bigl(P[w_k+\alpha_k d_k]\bigr)\;\le\; F_{\mathcal{S}_k}(w_k)+\sigma\,\widetilde{\nabla}F_{\mathcal{S}_k} (w_k)^T\bigl(P[w_k+\alpha_k d_k]-w_k\bigr),\qquad \sigma\in(0,1).`
- `slide25_formula1.png` — slide 25, formula 1: `g(w_{k+1})\approx g(w_k)+\nabla^2J(w_k)\,\Delta w_k .`
- `slide25_formula2.png` — slide 25, formula 2: `y_k\approx\nabla^2J(w_k)\,s_k .`
- `slide26_formula1.png` — slide 26, formula 1: `\alpha_k^{(1)}=\frac{s_k^\top y_k}{y_k^\top y_k} \quad(\min\|\alpha y-s\|^2),\qquad \alpha_k^{(2)}=\frac{s_k^\top s_k}{s_k^\top y_k} \quad(\min\|\alpha^{-1}s-y\|^2).`
- `slide29_formula1.png` — slide 29, formula 1: `J(w)=\frac{1}{N}\sum_{i=1}^N\Bigl[(w_1-a_i)^2+ 100\bigl((w_2-b_i)-(w_1-a_i)^2\bigr)^2\Bigr],`
- `slide34_formula1.png` — slide 34, formula 1: `g_k=\nabla J_{\mathcal S_k}(w_k)=\bigl[2\kappa\,(w_1+\bar\varepsilon_1),\, 2\,(w_2+\bar\varepsilon_2)\bigr]^{\!\top},\quad \bar\varepsilon_j=\frac{1}{n_k}\sum_{i\in\mathcal S_k}\varepsilon_j^{(i)}.`
