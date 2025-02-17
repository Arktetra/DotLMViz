<script lang="ts">
	import { page } from '$app/state';
	import { content } from '$lib/content';

	console.log(page.params);
	const title = $derived(page.params.title.split('-').join(' '));
	const description = $derived(
		content[page.params.title] ?? 'NO information available for this topic!!'
	);
</script>

<div
	class="my-10 min-h-[50vh] max-w-[90%] rounded-sm border bg-theme-g text-justify text-lg tracking-wider md:max-w-[80%]"
>
	<h1 class="bg-theme py-5 text-center font-main-a text-2xl font-extrabold uppercase text-theme-w">
		{title}
	</h1>
	<p class="my-5 px-10">
		{@html description}
	</p>
</div>
