document.addEventListener('DOMContentLoaded', function() {
    fetch('/data')
        .then(response => response.json())
        .then(data => {
            // Prepare pie chart data for today (latest entry)
            let todayData = data[data.length - 1];
            let todayPieData = [
                { label: "Focused", value: todayData.focused },
                { label: "Not Focused", value: todayData.not_focused }
            ];

            // Prepare pie chart data for weekly average
            let weeklyData = data.slice(-7);
            let weeklyFocused = d3.mean(weeklyData, d => d.focused);
            let weeklyNotFocused = 100 - weeklyFocused;
            let weeklyPieData = [
                { label: "Focused", value: weeklyFocused },
                { label: "Not Focused", value: weeklyNotFocused }
            ];

            // Draw pie charts
            drawPieChart('#todayChart', todayPieData);
            drawPieChart('#weeklyChart', weeklyPieData);

            // Draw line chart
            drawLineChart(data);

            // Home button click event
            document.getElementById('homeButton').addEventListener('click', function() {
                window.location.href = '/'; // Change this to your home page URL
            });
        });

    function drawPieChart(selector, data) {
        let width = 300, height = 300, radius = Math.min(width, height) / 2;
        let svg = d3.select(selector)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${width / 2}, ${height / 2})`);

        let color = d3.scaleOrdinal(['#4caf50', '#f44336']);
        let pie = d3.pie().value(d => d.value);
        let path = d3.arc().outerRadius(radius).innerRadius(0);

        let arc = svg.selectAll('arc')
            .data(pie(data))
            .enter()
            .append('g')
            .attr('class', 'arc');

        arc.append('path')
            .attr('d', path)
            .attr('fill', d => color(d.data.label));

        arc.append('text')
            .attr('transform', d => `translate(${path.centroid(d)})`)
            .attr('dy', '0.35em')
            .attr('text-anchor', 'middle')
            .text(d => `${d.data.label}: ${d.data.value}%`);
    }

    function drawLineChart(data) {
        let margin = { top: 20, right: 30, bottom: 30, left: 40 };
        let width = document.getElementById('lineChartContainer').clientWidth - margin.left - margin.right;
        let height = 400 - margin.top - margin.bottom;

        let svg = d3.select('#lineChart')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        let x = d3.scaleTime()
            .domain(d3.extent(data, d => new Date(d.date)))
            .range([0, width]);

        let y = d3.scaleLinear()
            .domain([0, 100])
            .range([height, 0]);

        let lineFocused = d3.line()
            .x(d => x(new Date(d.date)))
            .y(d => y(d.focused));

        let lineNotFocused = d3.line()
            .x(d => x(new Date(d.date)))
            .y(d => y(d.not_focused));

        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(d3.timeDay.every(1)).tickFormat(d3.timeFormat('%b %d')));

        svg.append('g')
            .call(d3.axisLeft(y));

        svg.append('path')
            .datum(data)
            .attr('fill', 'none')
            .attr('stroke', '#4caf50')
            .attr('stroke-width', 1.5)
            .attr('d', lineFocused);

        svg.append('path')
            .datum(data)
            .attr('fill', 'none')
            .attr('stroke', '#f44336')
            .attr('stroke-width', 1.5)
            .attr('d', lineNotFocused);

        svg.append('text')
            .attr('x', width - 6)
            .attr('y', y(data[0].focused))
            .attr('dy', '.35em')
            .attr('text-anchor', 'end')
            .style('fill', '#4caf50')
            .text('Focused');

        svg.append('text')
            .attr('x', width - 6)
            .attr('y', y(data[0].not_focused))
            .attr('dy', '.35em')
            .attr('text-anchor', 'end')
            .style('fill', '#f44336')
            .text('Not Focused');
    }
});