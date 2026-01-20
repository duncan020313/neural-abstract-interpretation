main(n)
{
  x = 0;
  y = 1;
  while (x < n)
  {
    x = x + 1;
    y = y * 2;
  }
  if (y == 32)
  {
    y = y + 1;
  }
  else
  {
    y = y - 1;
  }
  return y;
}
