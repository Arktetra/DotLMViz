<script lang="ts">
	import { page } from '$app/state';
	import { content } from '$lib/content';

	console.log(page.params);
	const title = $derived(page.params.title.split('-').join(' '));
	const description = $derived(content[page.params.title] ?? 'NO information available for this topic!!');
</script>

<div
	class="my-10 md:max-w-[80%] max-w-[90%] min-h-[50vh] bg-theme-g border rounded-sm text-justify text-lg tracking-wider"
>
	<h1 class="py-5 text-center font-main-a text-2xl font-extrabold uppercase text-theme-w bg-theme">
		{title}
	</h1>
	<p class="px-10 my-5">
		{@html description}
	</p>
</div>
