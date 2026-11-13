# فصل ۲ — مرور ادبیات (اسکلت + نقشه‌ی منابع، ۱۳ نوامبر ۲۰۲۶)

> هر بند باید پس از خواندن عمیق مقاله با قالب ۵ خطی (مسئله، روش، یافته، محدودیت، ربط به ما) پر شود. شناسه‌های arXiv برای پیدا کردن سریع آمده‌اند؛ قبل از نهایی شدن، مرجع کامل را از صفحه‌ی مقاله بردار.

## ۲-۱ پیش‌زمینه
- هم‌ترازی مدل‌های زبانی: SFT، RLHF/DPO، persona «دستیار».
- تنظیم دقیق و فرسایش ایمنی: Qi و همکاران (۲۰۲۳، تنظیم دقیق حتی با داده‌ی بی‌خطر ایمنی را تضعیف می‌کند).
- LoRA و rs-LoRA.

## ۲-۲ ناهم‌ترازی نوظهور: کشف و گستره
- Betley و همکاران — arXiv 2502.17424؛ Nature 2026 (کد ناامن → ناهم‌ترازی گسترده).
- Turner و همکاران — Model Organisms for EM، arXiv 2506.11613 (داده‌های متنی، مدل‌های کوچک، rank-1 LoRA).
- EM در یادگیری درون‌متنی — arXiv 2510.11288.
- EM در RL: Natural EM from reward hacking — arXiv 2511.18397؛ RL Can Amplify EM from Harmless Rewards (EMNLP 2026).
- Cross-Modal EM (EMNLP 2026)؛ Character as a Latent Variable — arXiv 2601.23081.
- Automated Subtle EM Research (اکتبر ۲۰۲۶).

## ۲-۳ نقد سنجش EM (اعتبار ارزیابی)
- EM as prompt sensitivity — arXiv 2507.06253.
- The devil in the details: format and coherence — arXiv 2511.20104.
- Overtrained, Not Misaligned — arXiv 2605.12199 (EM بعد از همگرایی وظیفه؛ توقف زودهنگام).
- What Shapes EM? training dynamics, priors, data — arXiv 2606.20814.
- نقدهای Rao و Schreiber (از فهرست خواندن پروپوزال).
→ نتیجه برای ما: دو داور، کنترل طول، آستانه‌ی ثبت‌شده، بازبینی انسانی.

## ۲-۴ سازوکار درونی EM — دیدگاه‌های رقیب
**دیدگاه persona/جهت خطی:**
- Wang و همکاران، Persona Features Control EM — arXiv 2506.19823 (ICLR 2026).
- Soligo و همکاران، Convergent Linear Representations؛ EM is Easy, Narrow is Hard — arXiv 2602.07852 (ICLR 2026).
- Nadaf، EM Recruits a Pre-existing Persona Subspace — arXiv 2607.21356.
- Misalignment Has a Personality (Big Five) — arXiv 2607.26389.
- Persona-Model Collapse — arXiv 2605.12850.
- Feature Superposition Geometry — arXiv 2605.00842.
- زیرفضای وزن‌ها: arXiv 2511.02022.
**دیدگاه مخالف:**
- Li و همکاران، EM Is Not Magical — arXiv 2608.29118 (EMNLP 2026): جهت عمومی قابل انتقال وجود ندارد؛ EM با تغییر persona فرق دارد.
→ این اختلاف، انگیزه‌ی مستقیم پ۲ است.

## ۲-۵ دفاع‌ها و مداخلات
**حین آموزش:**
- Kaczér و همکاران، In-Training Defenses — arXiv 2508.06249 (ICML 2026): KL، LDIFS، persona vectors، interleaving++، inoculation.
- Persona vectors / preventative steering — Chen و همکاران (۲۰۲۵).
- Inoculation prompting (دو مقاله‌ی فهرست پروپوزال).
- CAFT (Casademunt و همکاران، ۲۰۲۵).
- BLOCK-EM — arXiv 2602.00767 (ICML 2026): مسدودسازی ویژگی‌های SAE + **بازگشت ناهم‌ترازی در آموزش طولانی**.
- Method-conditional EM / persona-orthogonal fine-tuning — arXiv 2607.04510.
**پس از آموزش (بازهم‌ترازی):**
- Emergent re-alignment (Wang و همکاران)؛ Alignment Gating — arXiv 2606.09068.
**مقاومت در برابر دستکاری (پیوند با پ۳):**
- Tamper-resistant safeguards (TAR)؛ SEAM / Self-destructive LMs — arXiv 2505.12186 (ICLR 2026).

## ۲-۶ پنهان شدن به‌جای حذف
- Conditional misalignment — arXiv 2604.25891: interleaving و inoculation ناهم‌ترازی را پشت trigger پنهان می‌کنند.
- Consistency Training Can Entrench Misalignment — arXiv 2606.03810.
- BLOCK-EM §5 (rerouting)؛ Overtrained (رفتار پنهان با prompt خصمانه برمی‌گردد).
- پیوند با ادبیات unlearning: حمله‌های یادگیری مجدد (relearning attacks) که نشان می‌دهند «فراموشی» اغلب سطحی است.
- پیوند روان‌شناختی: savings (Ebbinghaus)، خاموشی و بازگشت خودبه‌خودی در شرطی‌سازی (spontaneous recovery، renewal).

## ۲-۷ جمع‌بندی و شکاف
جدول: هر دفاع × (ارزیابی استاندارد / آزمون عود / trigger / آموزش طولانی / سنجش درونی) — خانه‌های خالی = سهم این پایان‌نامه.
