document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll('.circle').forEach(circle => {
        const percent = circle.getAttribute('data-percent');
        const progress = circle.querySelector('circle.progress');
        const radius = progress.r.baseVal.value;
        const circumference = 2 * Math.PI * radius;

        progress.style.strokeDasharray = circumference;
        progress.style.strokeDashoffset = circumference - (circumference * percent / 100);
    });
});